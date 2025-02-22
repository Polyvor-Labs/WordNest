import torch
import torch.nn as nn
import torch.nn.functional as F

class MTPModule(nn.Module):
    """
    MTPModule implements a multi-task prediction (MTP) module using Transformer encoder layers.
    It takes two inputs—hidden states (h) and target embeddings (emb)—normalizes them, concatenates
    them, and projects the result to the model dimension before processing it through a stack of
    Transformer encoder layers.
    
    Attributes:
        depth (int): Number of Transformer encoder layers.
        layers (nn.ModuleList): A list of TransformerEncoderLayer modules.
        proj (nn.Linear): Linear projection to reduce concatenated features from 2*d_model to d_model.
    """
    def __init__(self, d_model=512, depth=1):
        """
        Initializes the MTPModule with the specified model dimension and depth.

        Args:
            d_model (int): Dimensionality of the model's hidden states (default: 512).
            depth (int): Number of Transformer encoder layers to apply (default: 1).
        """
        super().__init__()
        self.depth = depth
        # Create a stack of Transformer encoder layers with 8 attention heads each.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, 8) for _ in range(depth)])
        # Linear projection to combine and reduce concatenated hidden state and embedding from 2*d_model to d_model.
        self.proj = nn.Linear(2 * d_model, d_model)
        
    def forward(self, h, emb):
        """
        Processes the input hidden states and target embeddings through normalization, projection,
        and Transformer encoder layers to produce refined representations.

        Args:
            h (torch.Tensor): Hidden state tensor of shape (B, T, d_model) from the main network.
            emb (torch.Tensor): Embedding tensor of shape (B, T, d_model) corresponding to target tokens.

        Returns:
            torch.Tensor: The output tensor after multi-task prediction processing, with shape (B, T, d_model).
        """
        # Apply layer normalization to both the hidden states and the embeddings.
        h_norm = F.layer_norm(h, h.shape[-1:])
        emb_norm = F.layer_norm(emb, emb.shape[-1:])
        
        # Concatenate the normalized hidden states and embeddings along the last dimension.
        combined = torch.cat([h_norm, emb_norm], dim=-1)
        # Project the concatenated tensor back to the original model dimension.
        combined = self.proj(combined)
        
        # Pass the combined tensor through each Transformer encoder layer sequentially.
        for layer in self.layers:
            combined = layer(combined)
            
        return combined