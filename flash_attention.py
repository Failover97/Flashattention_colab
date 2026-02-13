import torch
import math

def pseudo_flash_attention(Q, K, V, mask=None, block_size_M=64, block_size_N=64):
    """
    Simulates FlashAttention using PyTorch blocks.
    
    Args:
        Q: Query tensor of shape (Batch, SeqLen_Q, HeadDim)
        K: Key tensor of shape (Batch, SeqLen_K, HeadDim)
        V: Value tensor of shape (Batch, SeqLen_K, HeadDim)
        mask: Optional boolean mask tensor of shape (SeqLen_Q, SeqLen_K). 
              False indicates "masked out" (ignore), True indicates "keep".
        block_size_M: Block size for Query sequence dimension (rows of O)
        block_size_N: Block size for Key/Value sequence dimension (cols of O)
        
    Returns:
        O: Output tensor of shape (Batch, SeqLen_Q, HeadDim)
    """
    
    # Batch size, Sequence Lengths, Head Dimension
    B, N_Q, D = Q.shape
    _, N_K, _ = K.shape
    
    # Set bool mask
    if mask is not None:
        assert mask.shape == (N_Q, N_K), f"Mask shape mismatch! Expected ({N_Q}, {N_K}), got {mask.shape}"
    else:
        mask = torch.ones((N_Q, N_K), device=Q.device, dtype=torch.bool)

    # Set mask to -inf where mask is False (masked out)
    mask = mask.to(torch.float32)
    mask = mask.masked_fill(mask == 0, -1e9)
        
    # Initialize Output in HBM
    O = torch.zeros_like(Q)
    
    # Initialize statistics for online softmax: l (sum of exps) and m (max exponent) in HBM
    # Shape: (B, N_Q, 1) - one value per query row
    l = torch.zeros((B, N_Q, 1), device=Q.device)
    m = torch.ones((B, N_Q, 1), device=Q.device) * float('-inf')
    
    # Scaling factor
    scale = 1.0 / math.sqrt(D)
    
    # Outer loop: Iterate over blocks of K and V (columns of the attention matrix)
    for j in range(0, N_K, block_size_N):
        # Define current block range for K and V
        j_end = min(j + block_size_N, N_K)
        
        # Load block of K and V from HBM to SRAM (simulated)
        # In real hardware: Load K[j:j_end] and V[j:j_end] into SRAM buffer reuse
        K_j = K[:, j:j_end, :]
        V_j = V[:, j:j_end, :]
        
        # Inner loop: Iterate over blocks of Q (rows of the attention matrix)
        for i in range(0, N_Q, block_size_M):
            # Define current block range for Q
            i_end = min(i + block_size_M, N_Q)
            
            # Load block of Q from HBM to SRAM (simulated)
            # In real hardware: Load Q[i:i_end] into SRAM buffer
            Q_i = Q[:, i:i_end, :]

            # Load block of mask from HBM to SRAM (simulated)
            # In real hardware: Load mask[i:i_end, j:j_end] into SRAM buffer
            if mask is not None:
                mask_block = mask[i:i_end, j:j_end]
            
            # Load previous global max and sum for this block row indices i:i_end
            m_prev = m[:, i:i_end, :]
            l_prev = l[:, i:i_end, :]
            O_i_prev = O[:, i:i_end, :]
            
            # =================================================================
            # TODO: Implement the FlashAttention core logic for this block pair
            # 1. Compute Q_i * K_j^T
            # 2. Apply scaling
            # 3. Add mask_block to S_ij
            # 4. Compute local max and local exp sum (Online Softmax)
            # 5. Update global max (m), global sum (l), and Output (O)
            # =================================================================
            
            # 1. Compute Score S_ij
            # S_ij shape should be (B, block_M, block_N)
            # Code:
            # =================================================================

            # =================================================================
            
            # 2. Apply Masking for S_ij
            # Code: 
            # =================================================================

            # =================================================================   
                    
            # 3. Compute local statistics for this block
            # m_block: max value in each row of S_ij (Shape: B, block_M, 1)
            # p_block: exp(S_ij - m_block) (Shape: B, block_M, block_N)
            # l_block: sum(p_block) (Shape: B, block_M, 1)
            # Code:
            # =================================================================

            # =================================================================
            
            # 4. Update global statistics (m, l) and Output (O)
            # Code:
            # =================================================================
            
            # =================================================================

            # Update O, l, m in the global memory
            # Code:
            # =================================================================

            # =================================================================
            
    # Final normalization of O by l
    O = O / l
    
    return O
