import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from .embedding import RotaryEmbedding, apply_rope

class MultiHeadLatentAttention(nn.Module):
    """
    MultiHeadLatentAttention implements a multi-head attention mechanism with latent compression
    and rotary positional embeddings. The attention is computed on a latent representation,
    where the key and value vectors are first compressed to a lower dimensionality before being
    projected back to the model dimension.
    
    Attributes:
        d_model (int): Dimensionality of the input and output features.
        n_heads (int): Number of attention heads.
        d_head (int): Dimensionality of each attention head (d_model / n_heads).
        d_kv_compress (int): Dimensionality to which keys and values are compressed.
    """
    def __init__(self, d_model=512, n_heads=8, d_kv_compress=64):
        """
        Initializes the MultiHeadLatentAttention module.
        
        Args:
            d_model (int): Input feature dimension (default: 512).
            n_heads (int): Number of attention heads (default: 8).
            d_kv_compress (int): Compression dimension for keys and values (default: 64).
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # Dimension per head
        self.d_kv_compress = d_kv_compress
        
        # Linear projections to compress input for key and value computation.
        self.w_dk = nn.Linear(d_model, d_kv_compress)
        self.w_dv = nn.Linear(d_model, d_kv_compress)
        
        # Linear projections to expand the compressed key and value back to model dimension.
        self.w_uk = nn.Linear(d_kv_compress, d_model)
        self.w_uv = nn.Linear(d_kv_compress, d_model)
        
        # Output linear projection.
        self.w_o = nn.Linear(d_model, d_model)
        
        # Rotary positional embedding initialization for each attention head.
        self.rotary = RotaryEmbedding(self.d_head)

    def forward(self, x):
        """
        Performs a forward pass through the multi-head latent attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model), where
                              B is the batch size,
                              T is the sequence length.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model) after applying multi-head attention.
        """
        B, T, _ = x.shape
        
        # Compress the input features for key and value computation.
        c_k = self.w_dk(x)  # Compressed key representation, shape: (B, T, d_kv_compress)
        c_v = self.w_dv(x)  # Compressed value representation, shape: (B, T, d_kv_compress)
        
        # Project the compressed representations back to the model dimension.
        k = self.w_uk(c_k)  # Key tensor, shape: (B, T, d_model)
        v = self.w_uv(c_v)  # Value tensor, shape: (B, T, d_model)
        
        # Compute queries directly from the input.
        q = rearrange(x, "b t (h d) -> b h t d", h=self.n_heads)  # Shape: (B, n_heads, T, d_head)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.n_heads)  # Shape: (B, n_heads, T, d_head)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.n_heads)  # Shape: (B, n_heads, T, d_head)
        
        # Generate rotary positional embeddings based on sequence length.
        freqs = self.rotary(T)
        
        # Apply rotary positional embeddings to query and key.
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        
        # Compute scaled dot-product attention scores.
        # 'bhid,bhjd->bhij' performs a dot product between the query and key.
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * (1.0 / np.sqrt(self.d_head))
        attn = torch.softmax(attn, dim=-1)  # Normalize attention scores
        
        # Apply attention weights to the value vectors.
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Rearrange the output back to (B, T, d_model)
        out = rearrange(out, "b h t d -> b t (h d)")
        
        # Final linear projection.
        return self.w_o(out)