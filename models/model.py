import torch
import torch.nn as nn
from .attention import MultiHeadLatentAttention
from .moe import WordnestMoE
from .mtp import MTPModule

class Wordnest(nn.Module):
    """
    Wordnest is a composite neural network model designed for language modeling tasks.
    It combines multiple components including an embedding layer, multi-head latent attention,
    mixture-of-experts (MoE) modules, and multi-task prediction (MTP) modules. The model
    processes input token sequences and generates predictions via a final linear output head.

    Attributes:
        embedding (nn.Embedding): Converts token indices to dense embeddings.
        layers (nn.ModuleList): A list of layers, each containing:
            - "attention": A multi-head latent attention module.
            - "moe": A Wordnest mixture-of-experts module.
        mtp (nn.ModuleList): A list of MTP modules for auxiliary prediction tasks.
        out_head (nn.Linear): Final linear layer that maps hidden representations to vocabulary logits.
    """
    def __init__(self, vocab_size=10000, d_model=512, n_layers=6, n_experts=8, mtp_depth=1):
        """
        Initializes the Wordnest model with embedding, attention, MoE, and MTP modules.

        Args:
            vocab_size (int): The size of the vocabulary (default: 10000).
            d_model (int): Dimensionality of the embeddings and model hidden state (default: 512).
            n_layers (int): Number of attention-MoE layers (default: 6).
            n_experts (int): Number of experts in the MoE module (default: 8).
            mtp_depth (int): Depth (number) of MTP modules (default: 1).
        """
        super().__init__()
        # Embedding layer to convert token indices into dense vector representations.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Create a stack of layers where each layer contains:
        #   - A multi-head latent attention module.
        #   - A mixture-of-experts (MoE) module.
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": MultiHeadLatentAttention(d_model),
                "moe": WordnestMoE(d_model, n_experts)
            })
            for _ in range(n_layers)
        ])
        
        # Initialize MTP modules to support additional multi-task predictions.
        self.mtp = nn.ModuleList([MTPModule(d_model, mtp_depth) for _ in range(mtp_depth)])
        
        # Final output head that projects the model's hidden states to the vocabulary logits.
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets=None):
        """
        Executes a forward pass through the Wordnest model.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (B, T),
                              where B is the batch size and T is the sequence length.
            targets (torch.Tensor, optional): Target token indices used for multi-task prediction.
                                              If provided, MTP outputs are computed based on shifted targets.

        Returns:
            tuple: A tuple containing:
                - main_output (torch.Tensor): Logits of shape (B, T, vocab_size) from the primary output head.
                - mtp_outputs (list): A list of logits from the MTP modules, each corresponding to different offsets.
        """
        # Convert input token indices to embeddings.
        x = self.embedding(x)
        
        # Pass the embeddings through each attention and MoE layer.
        for layer in self.layers:
            # Apply multi-head latent attention and add the residual connection.
            x = x + layer["attention"](x)
            # Process through the mixture-of-experts module.
            x = layer["moe"](x)
        
        mtp_outputs = []
        # If target tokens are provided, compute outputs from the MTP modules.
        if targets is not None:
            # Iterate over each MTP module with an increasing offset.
            for depth, mtp_layer in enumerate(self.mtp):
                offset = depth + 1
                if offset < x.size(1):
                    # Obtain target embeddings for a sequence shifted by the offset.
                    emb = self.embedding(targets[:, offset:offset + x.size(1)])
                    # Compute the MTP output using the corresponding segments of x and target embeddings.
                    mtp_out = mtp_layer(x[:, :-offset], emb[:, :x.size(1) - offset])
                    # Append the result after projecting it to vocabulary space.
                    mtp_outputs.append(self.out_head(mtp_out))
        
        # Compute the main output logits from the processed hidden states.
        main_output = self.out_head(x)
        return main_output, mtp_outputs