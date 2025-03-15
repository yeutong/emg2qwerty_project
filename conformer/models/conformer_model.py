import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ConvSubsampling(nn.Module):
    """CNN-based subsampling layer."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, stride=2, padding=1),  # Downsampling
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim * ((input_dim // 4) + 1), hidden_dim)  # Adjust for downsampling

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3).contiguous().flatten(2)  # Reshape for Transformer
        x = self.linear(x)
        return x

class ConformerBlock(nn.Module):
    """Conformer block with MHSA, convolution, and feedforward layers."""
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff1 = nn.Linear(hidden_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Self-Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))  # Residual connection

        # Convolutional Module
        conv_out = self.conv1(x.permute(1, 2, 0)).permute(2, 0, 1)  # 1D Conv
        x = self.norm2(x + self.dropout(conv_out))  # Residual connection

        # Feedforward Network
        ff_out = F.relu(self.ff1(x))
        ff_out = self.ff2(ff_out)
        x = x + self.dropout(ff_out)  # Residual connection

        return x

class ConformerModel(nn.Module):
    """Full Conformer Model for Keystroke Prediction."""
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=6, num_heads=4, ff_dim=256, dropout=0.1):
        super().__init__()
        self.subsampling = ConvSubsampling(input_dim, hidden_dim)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_blocks)
        ])
        self.fc_out = nn.Linear(hidden_dim, num_classes)  # Output layer for classification

    def forward(self, x):
        x = self.subsampling(x)
        x = x.permute(1, 0, 2)  # Required shape for Transformer: (seq_len, batch, feature_dim)
        for block in self.conformer_blocks:
            x = block(x)
        x = self.fc_out(x)
        return x

