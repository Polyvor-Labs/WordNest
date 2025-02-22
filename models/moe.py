import torch
import torch.nn as nn

class WordnestMoE(nn.Module):
    """
    WordnestMoE implements a Mixture-of-Experts (MoE) module that enhances the model's expressiveness
    by routing tokens through a set of specialized expert networks. Each token is processed by a shared expert
    and a weighted combination of top-k selected experts, determined via a gating mechanism.

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        n_experts (int): Total number of expert networks.
        k (int): Number of top experts to select per token.
        shared_expert (nn.Sequential): A shared feedforward network applied to all tokens.
        experts (nn.ModuleList): A list of individual expert networks.
        gate (nn.Linear): Linear layer that computes gating logits for expert selection.
        bias (nn.Parameter): Learnable bias added to the gating logits.
    """
    def __init__(self, d_model=512, n_experts=8, k=2):
        """
        Initializes the WordnestMoE module with a shared expert, multiple specialized experts,
        and a gating mechanism to select the top-k experts for each token.

        Args:
            d_model (int): Dimensionality of the model features (default: 512).
            n_experts (int): Number of expert networks available (default: 8).
            k (int): Number of experts to select for each token (default: 2).
        """
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        
        # Shared expert network applied to every token.
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # Define multiple expert networks; each is a feedforward network.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.SiLU(),
                nn.Linear(4 * d_model, d_model)
            )
            for _ in range(n_experts)
        ])
        
        # Gating network to determine the contribution of each expert.
        self.gate = nn.Linear(d_model, n_experts)
        # Learnable bias added to the gate logits.
        self.bias = nn.Parameter(torch.zeros(n_experts))

    def forward(self, x):
        """
        Processes the input tokens through the shared expert and the selected top-k experts.
        The outputs are aggregated with a residual connection from the original input.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, d_model), where B is the batch size and T is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, d_model) after applying MoE processing.
        """
        # Compute the output from the shared expert for all tokens.
        shared_out = self.shared_expert(x)
        B, T, _ = x.shape
        
        # Compute gating logits and add the learnable bias.
        logits = self.gate(x) + self.bias
        # Apply sigmoid to obtain gating scores in the range [0, 1].
        scores = torch.sigmoid(logits)
        
        # Select the top-k experts for each token based on the gating scores.
        topk_scores, topk_idx = torch.topk(scores, self.k, dim=-1)
        # Normalize the top-k scores to obtain proper weights.
        topk_scores = topk_scores.softmax(dim=-1)
        
        # Initialize tensor to accumulate expert outputs.
        expert_out = torch.zeros_like(x)
        
        # Loop over each expert and apply it to tokens for which it was selected.
        for i in range(self.n_experts):
            # Create a mask indicating tokens for which expert i is among the top-k selections.
            expert_mask = (topk_idx == i)
            if expert_mask.any():
                # Sum the gating scores for expert i across the k selections for each token.
                expert_scores = torch.where(expert_mask, topk_scores, torch.zeros_like(topk_scores)).sum(dim=-1)
                # Create a mask to select tokens that have expert i assigned.
                token_mask = expert_mask.any(dim=-1)
                # Retrieve tokens that are routed to expert i.
                selected_tokens = x[token_mask]
                
                if selected_tokens.size(0) > 0:
                    # Process the selected tokens through expert i.
                    expert_result = self.experts[i](selected_tokens)
                    # Weight the expert's output by its gating score and accumulate.
                    expert_out[token_mask] += expert_result * expert_scores[token_mask].unsqueeze(-1)
        
        # Return the final output as a sum of the original input, the shared expert output, and the aggregated expert outputs.
        return x + shared_out + expert_out