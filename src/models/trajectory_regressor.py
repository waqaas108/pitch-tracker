"""
Trajectory-to-metrics regression model.
Maps 2D trajectory sequences to velocity, break, and location.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class TrajectoryRegressor(nn.Module):
    """
    LSTM-based regressor that maps 2D trajectory to pitch metrics.
    
    Input: (batch, seq_len, 2) - 2D coordinates over time
    Output: (batch, 5) - [velocity, h_break, v_break, plate_x, plate_y]
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Output scaling (learned)
        self.output_scale = nn.Parameter(torch.tensor([
            100.0,  # velocity (mph)
            20.0,   # h_break (inches)
            20.0,   # v_break (inches)
            1.0,    # plate_x (meters)
            1.0,    # plate_y (meters)
        ]))
        
        self.output_bias = nn.Parameter(torch.tensor([
            85.0,   # velocity baseline
            0.0,    # h_break baseline
            0.0,    # v_break baseline
            0.0,    # plate_x baseline
            0.6,    # plate_y baseline (strike zone center)
        ]))
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, 2) trajectory coordinates
            lengths: (batch,) actual sequence lengths (for masking)
            
        Returns:
            (batch, 5) predicted metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Handle NaN values (missed detections) - replace with interpolated
        x = self._interpolate_missing(x)
        
        # Normalize input coordinates
        x = self._normalize_coords(x)
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Use final hidden states from both directions
        h_forward = h_n[-2]  # last layer, forward
        h_backward = h_n[-1]  # last layer, backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Output
        out = self.output_head(h_combined)
        
        # Scale and shift
        out = out * self.output_scale + self.output_bias
        
        return out
    
    def _normalize_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize 2D coordinates to roughly [-1, 1] range."""
        # Assume 1920x1080 image
        x = x.clone()
        x[..., 0] = (x[..., 0] - 960) / 960
        x[..., 1] = (x[..., 1] - 540) / 540
        return x
    
    def _interpolate_missing(self, x: torch.Tensor) -> torch.Tensor:
        """Linear interpolation for NaN values."""
        x = x.clone()
        for b in range(x.shape[0]):
            for d in range(x.shape[2]):
                vals = x[b, :, d]
                mask = torch.isnan(vals)
                if mask.any() and not mask.all():
                    indices = torch.arange(len(vals), device=x.device, dtype=torch.float)
                    valid_idx = indices[~mask]
                    valid_vals = vals[~mask]
                    interp_vals = torch.interp(indices[mask], valid_idx, valid_vals)
                    x[b, mask, d] = interp_vals
        return x


class TrajectoryRegressorConv(nn.Module):
    """
    1D-Conv alternative to LSTM for trajectory regression.
    Often faster and works well for fixed-length sequences.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 5,
        max_seq_len: int = 50
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.output_scale = nn.Parameter(torch.tensor([100.0, 20.0, 20.0, 1.0, 1.0]))
        self.output_bias = nn.Parameter(torch.tensor([85.0, 0.0, 0.0, 0.0, 0.6]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, 2) -> (batch, 2, seq)
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x * self.output_scale + self.output_bias
