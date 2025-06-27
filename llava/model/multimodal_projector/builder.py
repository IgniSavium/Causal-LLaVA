import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

# Cross-Attention Module for Causal Inference
class ConfounderCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        # Linear projection for key and value
        self.kv_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.size()

        # Project key and value
        key = self.kv_proj(key)
        value = self.kv_proj(value)

        # Reshape and compute attention scores
        q = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key.view(batch_size, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply attention mask if provided
        if attention_mask is not None:
            scores += attention_mask

        # Compute attention weights and output
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return attn_output


class CausalInterventionProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        # Cross-attention for visual confounders
        self.visual_cross_attn = ConfounderCrossAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        # Predefined visual confounders (e.g. 80 classes in MSCOCO)
        self.visual_confounders = nn.Parameter(torch.zeros(80, config.hidden_size), requires_grad=False)

    def forward(self, x):
        # Compute MLP output
        mlp_output = self.mlp(x)

        # Expand visual confounders to match batch size
        batch_size = x.size(0)
        visual_confounders_expanded = self.visual_confounders.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention on visual confounders
        visual_attn_output = self.visual_cross_attn(
            query=mlp_output,
            key=visual_confounders_expanded,
            value=visual_confounders_expanded
        )

        # Combine MLP output with visual attention output
        return mlp_output + visual_attn_output

    def forward_mlp_only(self, x):
        # Only compute MLP part
        return self.mlp(x)


class CausalInterventionLinearProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Linear layer for feature transformation
        self.linear = nn.Linear(config.mm_hidden_size, config.hidden_size)
        # Cross-attention for visual confounders
        self.visual_cross_attn = ConfounderCrossAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads
        )
        # Predefined visual confounders (80 classes in MSCOCO)
        self.visual_confounders = nn.Parameter(torch.zeros(80, config.hidden_size), requires_grad=False)

    def forward(self, x):
        # Compute linear output
        linear_output = self.linear(x)

        # Expand visual confounders to match batch size
        batch_size = x.size(0)
        visual_confounders_expanded = self.visual_confounders.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention on visual confounders
        visual_attn_output = self.visual_cross_attn(
            query=linear_output,
            key=visual_confounders_expanded,
            value=visual_confounders_expanded
        )

        # Combine linear output with visual attention output
        return linear_output + visual_attn_output

    def forward_linear_only(self, x):
        # Only compute linear part
        return self.linear(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        # Return the new CausalInterventionLinearProjector for "linear" projector_type
        return CausalInterventionLinearProjector(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        if mlp_depth == 2 and projector_type == "mlp2x_gelu":
            # Return the new CausalInterventionProjector for "mlp2x_gelu" projector_type
            return CausalInterventionProjector(config)
        else:
            # Original logic (no causal intervention) for other projector_types
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
