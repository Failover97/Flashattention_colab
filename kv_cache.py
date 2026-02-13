import torch

class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, head_dim, dtype=torch.float32, device='cpu'):
        """
        Initializes the Key-Value Cache.
        
        Args:
            max_batch_size: Maximum batch size to support.
            max_seq_len: Maximum sequence length to support.
            n_heads: Number of attention heads.
            head_dim: Dimension of each attention head.
            dtype: Data type for the cache.
            device: Device to store the cache on.
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.current_seq_len = 0
        
        # Pre-allocate memory for Keys and Values
        # Shape: (Batch, N_Heads, Seq_Len, Head_Dim) or (Batch, Seq_Len, N_Heads, Head_Dim)
        # Let's use standard (Batch, N_Heads, Seq_Len, Head_Dim) for this assignment
        self.k_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        self.v_cache = torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        
    def update(self, new_k, new_v):
        """
        Updates the cache with new key/value states for the current step.
        
        Args:
            new_k: New keys for the current token. Shape: (Batch, N_Heads, L_new, Head_Dim)
            new_v: New values for the current token. Shape: (Batch, N_Heads, L_new, Head_Dim)
            
        Returns:
            keys: The full keys up to the current position. Shape: (Batch, N_Heads, Current_Seq_Len, Head_Dim)
            values: The full values up to the current position. Shape: (Batch, N_Heads, Current_Seq_Len, Head_Dim)
        """
        # =================================================================
        # TODO: Implement Cache Update
        # 1. Determine the length of the new sequence segment (L_new).
        #    - new_k shape is (Batch, N_Heads, L_new, Head_Dim). Usually L_new=1 during decoding.
        #
        # 2. Check if there is enough space in the cache.
        #    - If current_seq_len + L_new > max_seq_len, raise an error or handle overflow (raising error is fine for this HW).
        #
        # 3. Store new_k and new_v into self.k_cache and self.v_cache.
        #    - You need to insert them at [:, :, self.current_seq_len : self.current_seq_len + L_new, :]
        #
        # 4. Update the internal counter: self.current_seq_len += L_new
        #
        # 5. Return the valid part of the cache.
        #    - Slice k_cache and v_cache to get [:, :, :self.current_seq_len, :]
        # =================================================================
        assert new_k.shape == new_v.shape, "new_k and new_v must have the same shape"
        l_new = new_k.shape[2]

        # Use tensor _copy function to copy new_k and new_v to the cache
        # =================================================================
        # Code:
        if self.current_seq_len + l_new > self.max_seq_len:
          raise ValueError("Cache overflow!")
        self.k_cache[:, :, self.current_seq_len:self.current_seq_len + l_new, :].copy_(new_k)
        self.v_cache[:, :, self.current_seq_len:self.current_seq_len + l_new, :].copy_(new_v)
        self.current_seq_len += l_new
        keys = self.k_cache[:, :, :self.current_seq_len, :]
        values = self.v_cache[:, :, :self.current_seq_len, :]

        # =================================================================

        return keys, values

    def reset(self):
        """Resets the cache counter."""
        self.current_seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

