# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from emg2qwerty.modules import SpectrogramNorm, MultiBandRotationInvariantMLP, TDSConvEncoder
from emg2qwerty.charset import charset

class TDSConvCTC(nn.Module):
    def __init__(
        self,
        in_features=528,  # 33 frequency bins * 16 channels per band
        mlp_features=[384],
        block_channels=[24, 24, 24, 24],
        kernel_width=32,
        num_bands=2,
        electrode_channels=16,
    ):
        super().__init__()
        
        num_features = num_bands * mlp_features[-1]
        
        # Model components directly matching their Lightning module
        self.spec_norm = SpectrogramNorm(channels=num_bands * electrode_channels)
        
        self.rotation_mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=num_bands
        )
        
        self.flatten = nn.Flatten(start_dim=2)
        
        self.tds_encoder = TDSConvEncoder(
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width
        )
        
        self.output_proj = nn.Linear(num_features, charset().num_classes)
    
    def forward(self, inputs, input_lengths=None):
        # Apply spectrogram normalization
        x = self.spec_norm(inputs)  # (T, N, bands=2, C=16, freq)
        
        # Process with rotation-invariant MLP
        x = self.rotation_mlp(x)  # (T, N, bands=2, mlp_features[-1])
        
        # Flatten bands dimension
        x = self.flatten(x)  # (T, N, num_features)
        
        # Process with TDS encoder
        x = self.tds_encoder(x)  # (T, N, num_features)
        
        # Project to output classes
        x = self.output_proj(x)  # (T, N, num_classes)
        
        return x
    


class EMGCNNBiLSTM(nn.Module):
    def __init__(self, in_features=1056, num_classes=99, dropout=0.3):
        super().__init__()

        # Reshape and initial feature extraction
        # Input is (T, N, 2, 16, 33) -> flatten to (T, N, 1056)
        # Simpler CNN feature extractor - process time dimension
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_features, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # BiLSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=False,
            dropout=dropout,
            bidirectional=True
        )

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, input_lengths=None):
        # x shape: [T, N, B, C, F] - [time, batch, bands, channels, freq]
        T, N, B, C, F = x.shape

        # Flatten all features: [T, N, B*C*F]
        x = x.reshape(T, N, B*C*F)

        # To format expected by Conv1d: [N, B*C*F, T]
        x = x.permute(1, 2, 0)

        # Apply CNN layers
        x = self.conv_layers(x)

        # To format expected by LSTM: [T', N, features]
        x = x.permute(2, 0, 1)

        # Adjust sequence lengths for LSTM if provided
        if input_lengths is not None:
            # Account for pooling (divided by 4 due to 2 pooling layers)
            lstm_lengths = torch.div(input_lengths, 4, rounding_mode='floor')
            lstm_lengths = torch.clamp(lstm_lengths, min=1)

            # Pack padded sequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lstm_lengths.cpu(), enforce_sorted=False
            )

            # Process with LSTM
            packed_output, _ = self.lstm(packed_x)

            # Unpack
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        else:
            # Process without packing
            x, _ = self.lstm(x)

        # Apply classifier
        time_steps, batch_size, hidden_dim = x.size()
        x = x.reshape(-1, hidden_dim)
        x = self.classifier(x)
        x = x.view(time_steps, batch_size, -1)

        return x
