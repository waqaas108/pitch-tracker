"""
Pitch type classification model.
Classifies pitch type from trajectory features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PitchClassifier(nn.Module):
    """
    Classifies pitch type from trajectory and/or predicted metrics.
    
    Can operate in two modes:
    1. From raw trajectory (2D coords over time)
    2. From predicted metrics (velocity, break, etc.)
    """
    
    def __init__(
        self,
        num_classes: int = 9,
        trajectory_dim: int = 2,
        metrics_dim: int = 5,
        hidden_dim: int = 64,
        use_trajectory: bool = True,
        use_metrics: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.use_trajectory = use_trajectory
        self.use_metrics = use_metrics
        self.num_classes = num_classes
        
        # Trajectory encoder (if used)
        if use_trajectory:
            self.traj_encoder = nn.LSTM(
                input_size=trajectory_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            traj_out_dim = hidden_dim * 2
        else:
            traj_out_dim = 0
        
        # Metrics encoder (if used)
        if use_metrics:
            self.metrics_encoder = nn.Sequential(
                nn.Linear(metrics_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            metrics_out_dim = hidden_dim
        else:
            metrics_out_dim = 0
        
        # Combined classifier
        combined_dim = traj_out_dim + metrics_out_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        trajectory: Optional[torch.Tensor] = None,
        metrics: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            trajectory: (batch, seq_len, 2) 2D coordinates
            metrics: (batch, 5) predicted metrics [velo, h_break, v_break, x, y]
            
        Returns:
            (batch, num_classes) logits
        """
        features = []
        
        if self.use_trajectory and trajectory is not None:
            # Normalize trajectory
            traj = self._normalize_trajectory(trajectory)
            _, (h_n, _) = self.traj_encoder(traj)
            # Concat forward and backward final states
            traj_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
            features.append(traj_feat)
        
        if self.use_metrics and metrics is not None:
            # Normalize metrics
            metrics_norm = self._normalize_metrics(metrics)
            metrics_feat = self.metrics_encoder(metrics_norm)
            features.append(metrics_feat)
        
        if not features:
            raise ValueError("At least one of trajectory or metrics must be provided")
        
        combined = torch.cat(features, dim=1)
        logits = self.classifier(combined)
        
        return logits
    
    def _normalize_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize trajectory coordinates."""
        x = x.clone()
        x[..., 0] = (x[..., 0] - 960) / 960
        x[..., 1] = (x[..., 1] - 540) / 540
        # Replace NaN with 0 (will be handled by LSTM)
        x = torch.nan_to_num(x, 0.0)
        return x
    
    def _normalize_metrics(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize metrics to roughly unit scale."""
        x = x.clone()
        # [velocity, h_break, v_break, plate_x, plate_y]
        scales = torch.tensor([100.0, 20.0, 20.0, 1.0, 1.0], device=x.device)
        offsets = torch.tensor([85.0, 0.0, 0.0, 0.0, 0.6], device=x.device)
        return (x - offsets) / scales
    
    def predict(self, trajectory: Optional[torch.Tensor] = None,
                metrics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(trajectory, metrics)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, trajectory: Optional[torch.Tensor] = None,
                      metrics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(trajectory, metrics)
        return F.softmax(logits, dim=1)


class MetricsOnlyClassifier(nn.Module):
    """
    Simple MLP classifier that only uses predicted metrics.
    Useful as a baseline or when trajectory quality is poor.
    """
    
    def __init__(self, num_classes: int = 9, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        # Normalize
        scales = torch.tensor([100.0, 20.0, 20.0, 1.0, 1.0], device=metrics.device)
        offsets = torch.tensor([85.0, 0.0, 0.0, 0.0, 0.6], device=metrics.device)
        x = (metrics - offsets) / scales
        return self.net(x)
