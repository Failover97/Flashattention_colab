import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from flash_attention import pseudo_flash_attention
from kv_cache import KVCache
import types

def load_models(base_model_id, lora_model_id=None):
    """
    Loads the base model and optionally the LoRA adapter.
    
    Args:
        base_model_id: HuggingFace ID for the base model (e.g. 'gpt2')
        lora_model_id: HuggingFace ID for the LoRA adapter (optional)
        
    Returns:
        model: The loaded model (with LoRA if provided)
        tokenizer: The loaded tokenizer
    """
    print(f"Loading base model: {base_model_id}...")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # =================================================================
    # TODO: Load Base Model and LoRA Adapter
    # 1. Load the base model using AutoModelForCausalLM
    # 2. If lora_model_id is provided:
    #    a. Load the LoRA adapter using PeftModel.from_pretrained
    #    b. Merge the adapter weights into the base model (optional, but good for inference speed) 
    #       or just return the PeftModel which wraps the base model.
    #       For this assignment, returning the wrapped PeftModel is fine.
    # =================================================================
    # Code:
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    if lora_model_id is not None:
      model = PeftModel.from_pretrained(model, lora_model_id)
      model = model.merge_and_unload()
    # =================================================================
    
    return model, tokenizer

def custom_opt_attention_forward(self, hidden_states, **kwargs):
    """
    Custom forward pass for OPTAttention that uses Pseudo FlashAttention and KVCache.
    This method will replace the original 'forward' method of the attention layer.
    """
    # self is the OPTAttention instance
    # Dimensions
    batch_size, seq_len, _ = hidden_states.shape
    num_heads = self.num_heads
    head_dim = self.head_dim
    
    # 1. Projections
    # OPT stores weights in q_proj, k_proj, v_proj, out_proj
    # Note: OPT's q_proj output shape is (Batch, Seq, Num_Heads * Head_Dim)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # Reshape to (Batch, Num_Heads, Seq_Len, Head_Dim) for Cache Update
    # Transpose: (Batch, Seq, Num_Heads, Head_Dim) -> (Batch, Num_Heads, Seq, Head_Dim)
    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # =================================================================
    # TODO: Use KV Cache and Pseudo FlashAttention
    # 1. Access the cache attached to this layer (self.my_kv_cache)
    #    Note: We attached 'my_kv_cache' in the 'enable_custom_kernels' function.
    # 2. Update the cache with the new key/value states.
    #    - If it's the first step (prefill), we update with the full sequence.
    #    - If it's a decoding step, we update with the current token.
    #    Hint: self.my_kv_cache.update(...)
    # 3. Retrieve the full keys and values from the cache.
    # 4. Prepare Q, K, V for Pseudo FlashAttention.
    #    - Pseudo FlashAttention expects (Batch, Seq, Head_Dim).
    #    - You need to merge Batch and Num_Heads dimensions: (Batch * Num_Heads, Seq, Head_Dim)
    # 5. Create a Causal Mask (since this is autoregressive generation).
    #    - Shape: (Seq_Len, Seq_Len)
    #    - Lower triangular matrix of ones.
    # 6. Call pseudo_flash_attention(Q, K, V, mask=causal_mask)
    # 7. Reshape output back to (Batch, Seq, Num_Heads * Head_Dim)
    # =================================================================
    
    # Placeholder for student implementation
    # 1. Update Cache
    # k_full, v_full = self.my_kv_cache.update(...)
    k_full, v_full = self.my_kv_cache.update(key_states, value_states)
    k_full = k_full[:batch_size]
    v_full = v_full[:batch_size]
    
    # 2. Reshape for FlashAttention
    # Q_reshaped = ...
    # K_reshaped = ...
    # V_reshaped = ...

    total_seq_len = k_full.shape[2]
    Q_reshaped = query_states.reshape(batch_size * num_heads, seq_len, head_dim)
    K_reshaped = k_full.reshape(batch_size * num_heads, total_seq_len, head_dim)
    V_reshaped = v_full.reshape(batch_size * num_heads, total_seq_len, head_dim)
    
    # 3. Create Causal Mask
    # mask = 
    mask = torch.zeros((seq_len, total_seq_len), device=hidden_states.device, dtype=torch.bool)
    for i in range(seq_len):
      mask[i, :total_seq_len - seq_len + i + 1] = True
    
    # 4. Call FlashAttention
    # attn_output = pseudo_flash_attention(Q_reshaped, K_reshaped, V_reshaped, mask=mask)
    attn_output = pseudo_flash_attention(Q_reshaped.float(), K_reshaped.float(), V_reshaped.float(), mask=mask)
    attn_output = attn_output.to(hidden_states.dtype)
    
    # 5. Reshape back
    # attn_output = ...
    attn_output = attn_output.reshape(batch_size, num_heads, seq_len, head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, num_heads * head_dim)

    # Final projection
    attn_output = self.out_proj(attn_output)
    
    return attn_output, None


def enable_custom_kernels(model):
    """
    Replaces the attention forward pass of the model with the custom one.
    Also initializes and attaches a KVCache to each layer.
    """
    print("Enabling custom kernels (Pseudo FlashAttention + KVCache)...")
    
    # Assuming OPT structure: model.model.decoder.layers
    # Adjust path if using a different model (e.g. GPT2)
    if hasattr(model, 'base_model'):
        # It's a PeftModel
        layers = model.base_model.decoder.layers
        config = model.base_model.config
    else:
        # It's a raw HF model
        layers = model.model.decoder.layers
        config = model.config

    max_batch_size = 4 # Fixed for this assignment
    max_seq_len = 128  # Fixed for this assignment
    
    for i, layer in enumerate(layers):
        # Access the self-attention module
        # For OPT, it's 'self_attn'
        attn_module = layer.self_attn
        
        # Initialize Custom KV Cache for this layer
        # Attach it to the module so we can access it in 'forward'
        attn_module.my_kv_cache = KVCache(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            n_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=model.device,
            dtype=model.dtype
        )
        
        # Monkey patch the forward method
        # We bind the custom function to the instance
        attn_module.forward = types.MethodType(custom_opt_attention_forward, attn_module)
        
    print("Custom kernels enabled.")


def generate_text(model, tokenizer, prompt, max_new_tokens=10, device='cpu', use_custom=False):
    """
    Generates text using the model.
    """
    model.to(device)
    
    if use_custom:
        enable_custom_kernels(model)
        # Note: When using custom kernels with manual KV Cache management, 
        # standard model.generate might conflict if it tries to pass past_key_values.
        # But since our custom forward ignores 'past_key_value' argument and uses 'self.my_kv_cache',
        # it might work if we set use_cache=False in generate to avoid passing internal cache.
    
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # We set use_cache=False to prevent HF from maintaining its own KV cache tuple,
        # because we are maintaining our own in 'self.my_kv_cache'.
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2, # Penalize repetition
            no_repeat_ngram_size=3, # Prevent repeating 3-grams
            use_cache=False if use_custom else True 
        )
        
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
