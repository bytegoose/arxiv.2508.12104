"""model.py

This module implements the Model class based on a decoder-only transformer
with Qwen2-inspired modifications. It uses PyTorch and builds on components such as:
 - Token Embedding
 - Rotary Positional Embeddings
 - Transformer Blocks with Pre-Layer Normalization,
   Grouped-Query Self-Attention, and Feed-Forward Networks with SwiGLU activations.
 - Output projection tied with the token embedding weights.
 
The Model class provides:
   - forward(): computes logits for a batch of input token sequences.
   - generate(): autoregressively generates token sequences given a prompt.
   
All hyperparameters are retrieved from a configuration dictionary (e.g., from config.yaml).
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Helper functions for rotary embeddings
# ---------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Splits the last dimension into two halves and rotates them.
    Args:
        x (torch.Tensor): Input tensor of shape (..., dim)
    Returns:
        torch.Tensor: Rotated tensor of the same shape.
    """
    x_dim = x.shape[-1]
    assert x_dim % 2 == 0, "Dimension must be even for rotary embeddings."
    x1 = x[..., : x_dim // 2]
    x2 = x[..., x_dim // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to tensor x.
    Args:
        x (torch.Tensor): Input tensor of shape [B, T, num_heads, head_dim]
        cos (torch.Tensor): Cosine embedding of shape [1, T, 1, head_dim]
        sin (torch.Tensor): Sine embedding of shape [1, T, 1, head_dim]
    Returns:
        torch.Tensor: Tensor with rotary positional embeddings applied.
    """
    return x * cos + rotate_half(x) * sin


def get_rotary_embeddings(seq_len: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotary positional cosine and sine embeddings.
    Args:
        seq_len (int): Sequence length.
        head_dim (int): Dimension of each attention head.
        device (torch.device): Torch device.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: cos and sin embeddings with shapes [1, seq_len, 1, head_dim]
    """
    # Generate frequencies based on head dimension (even dimension assumed)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)  # [seq_len, head_dim/2]
    # Duplicate to match head_dim
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, head_dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
    return cos, sin

# ---------------------------
# Self-Attention Module with Grouped-Query Attention
# ---------------------------
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, group_size: int = 1) -> None:
        """
        Self-Attention module with support for grouped-query attention.
        Args:
            hidden_dim (int): Model hidden dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            group_size (int): Group size for grouped-query attention; default 1 implies standard attention.
        """
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_heads: int = num_heads
        self.group_size: int = group_size
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads."
        self.head_dim: int = hidden_dim // num_heads
        self.scale: float = math.sqrt(self.head_dim)
        self.q_proj: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute self-attention.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, hidden_dim].
        Returns:
            torch.Tensor: Output tensor of shape [B, T, hidden_dim].
        """
        B, T, _ = x.size()
        # Linear projections
        q = self.q_proj(x)  # [B, T, hidden_dim]
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Reshape into multiple heads
        q = q.view(B, T, self.num_heads, self.head_dim)  # [B, T, num_heads, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        
        # Apply rotary positional embeddings
        device = x.device
        cos, sin = get_rotary_embeddings(T, self.head_dim, device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # If grouped-query attention is enabled and valid, perform grouping.
        if self.group_size > 1 and self.num_heads % self.group_size == 0:
            num_groups: int = self.num_heads // self.group_size
            # Reshape to include grouping: [B, T, num_groups, group_size, head_dim]
            q = q.view(B, T, num_groups, self.group_size, self.head_dim)
            k = k.view(B, T, num_groups, self.group_size, self.head_dim)
            v = v.view(B, T, num_groups, self.group_size, self.head_dim)
            # For simplicity, we then flatten back to the standard multi-head shape.
            q = q.view(B, T, self.num_heads, self.head_dim)
            k = k.view(B, T, self.num_heads, self.head_dim)
            v = v.view(B, T, self.num_heads, self.head_dim)
        
        # Permute for attention computation: [B, num_heads, T, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, num_heads, T, T]
        # Causal mask: lower triangular mask to prevent attending to future tokens.
        causal_mask = torch.tril(torch.ones((T, T), device=device)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)  # [B, num_heads, T, head_dim]
        # Reshape back: [B, T, hidden_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, self.hidden_dim)
        output = self.out_proj(attn_output)
        return output

# ---------------------------
# Feed-Forward Network with SwiGLU Activation
# ---------------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        """
        Feed-forward network with SwiGLU activation.
        This network consists of:
          - A linear expansion from hidden_dim to 2 * mlp_dim (for splitting).
          - SwiGLU activation: split into two halves, apply SiLU and multiply element-wise.
          - A linear projection back to hidden_dim.
        Args:
            hidden_dim (int): Input and output dimension.
            mlp_dim (int): Intermediate expansion dimension.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.fc1: nn.Linear = nn.Linear(hidden_dim, mlp_dim * 2)
        self.fc2: nn.Linear = nn.Linear(mlp_dim, hidden_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, hidden_dim].
        Returns:
            torch.Tensor: Output tensor of shape [B, T, hidden_dim].
        """
        x_fc = self.fc1(x)  # [B, T, mlp_dim*2]
        x1, x2 = x_fc.chunk(2, dim=-1)  # Each of shape [B, T, mlp_dim]
        x_act = x1 * F.silu(x2)
        x_act = self.dropout(x_act)
        output = self.fc2(x_act)
        return output

# ---------------------------
# Transformer Block
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0, group_size: int = 1) -> None:
        """
        Transformer block consisting of:
         - Pre-layer normalization.
         - Self-attention with grouped-query attention.
         - Residual connection.
         - Second layer normalization.
         - Feed-forward network with SwiGLU activation.
         - Residual connection.
        Args:
            hidden_dim (int): Model hidden dimension.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Feed-forward network expansion dimension.
            dropout (float): Dropout probability.
            group_size (int): Group size for grouped-query attention.
        """
        super().__init__()
        self.ln1: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.attn: SelfAttention = SelfAttention(hidden_dim, num_heads, dropout, group_size)
        self.ln2: nn.LayerNorm = nn.LayerNorm(hidden_dim)
        self.ff: FeedForward = FeedForward(hidden_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a transformer block.
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, hidden_dim].
        Returns:
            torch.Tensor: Output tensor of shape [B, T, hidden_dim].
        """
        attn_out = self.attn(self.ln1(x))
        x = x + attn_out
        ff_out = self.ff(self.ln2(x))
        x = x + ff_out
        return x

# ---------------------------
# Model Class (Decoder-Only Transformer)
# ---------------------------
class Model(nn.Module):
    def __init__(self, model_config: Dict[str, any]) -> None:
        """
        Initialize the decoder-only transformer model.
        Reads configuration from model_config and supports variants:
          COMET-S, COMET-M, and COMET-L.
        Args:
            model_config (Dict[str, any]): Configuration dictionary with keys:
                - "variant": string, one of "COMET-S", "COMET-M", "COMET-L" (default "COMET-S").
                - "context_window": int, context window size (default 8192).
                - "vocab_size": int, vocabulary size (default 7105).
                - "dropout": float, dropout probability (default 0.0).
                - "group_size": int, group size for attention (default 1).
        """
        super().__init__()
        variant: str = model_config.get("variant", "COMET-S")
        variants: Dict[str, Dict[str, int]] = {
            "COMET-S": {"layers": 6, "hidden_dimension": 768, "heads": 12, "mlp_dimension": 3072},
            "COMET-M": {"layers": 12, "hidden_dimension": 768, "heads": 12, "mlp_dimension": 3072},
            "COMET-L": {"layers": 16, "hidden_dimension": 2048, "heads": 32, "mlp_dimension": 8192},
        }
        variant_params: Dict[str, int] = variants.get(variant, variants["COMET-S"])
        self.num_layers: int = variant_params["layers"]
        self.hidden_dim: int = variant_params["hidden_dimension"]
        self.num_heads: int = variant_params["heads"]
        self.mlp_dim: int = variant_params["mlp_dimension"]
        self.context_window: int = model_config.get("context_window", 8192)
        self.vocab_size: int = model_config.get("vocab_size", 7105)
        dropout: float = model_config.get("dropout", 0.0)
        group_size: int = model_config.get("group_size", 1)

        # Token embedding layer
        self.token_embedding: nn.Embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        
        # Stack of transformer blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.num_heads, self.mlp_dim, dropout, group_size)
            for _ in range(self.num_layers)
        ])
        
        # Final layer normalization
        self.final_ln: nn.LayerNorm = nn.LayerNorm(self.hidden_dim)
        
        # Output projection layer (weight-tied to token_embedding)
        self.output_proj: nn.Linear = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        Args:
            input_ids (torch.Tensor): Tensor of shape [batch_size, seq_length] of token IDs.
        Returns:
            torch.Tensor: Logits of shape [batch_size, seq_length, vocab_size].
        """
        # Embedding lookup
        embeddings: torch.Tensor = self.token_embedding(input_ids)  # [B, T, hidden_dim]
        x: torch.Tensor = embeddings
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        logits: torch.Tensor = self.output_proj(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt: List[int], max_tokens: int, temperature: float = 1.0) -> List[int]:
        """
        Autoregressive generation using Monte Carlo sampling.
        Given a prompt (list of token IDs), generate up to max_tokens tokens.
        Args:
            prompt (List[int]): List of token IDs as the prompt.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling (default 1.0).
        Returns:
            List[int]: Generated token sequence (prompt concatenated with new tokens).
        """
        self.eval()
        device: torch.device = next(self.parameters()).device
        generated: List[int] = prompt.copy()
        for _ in range(max_tokens):
            # Ensure prompt length within context window
            input_seq: List[int] = generated[-self.context_window :]
            input_ids: torch.Tensor = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
            logits: torch.Tensor = self.forward(input_ids)  # [1, seq_len, vocab_size]
            next_logits: torch.Tensor = logits[:, -1, :]  # [1, vocab_size]
            next_logits = next_logits / temperature
            probs: torch.Tensor = F.softmax(next_logits, dim=-1)  # [1, vocab_size]
            next_token: int = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            # Optional early stopping can be added based on special EOS tokens if desired.
        return generated
