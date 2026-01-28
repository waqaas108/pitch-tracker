"""
Physics-constrained trajectory fitting.
Optimizes pitch parameters (velocity, spin) to match observed 2D/3D measurements.
This is the core of the new approach - fit physics, don't regress directly.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from typing import Tuple, Optional, List
import torch
import torch.nn as nn

from .trajectory import (
    PitchParams, PitchResult, simulate_pitch,
    mph_to_ms, ms_to_mph, rpm_to_rads,
    MOUND_TO_PLATE, RELEASE_HEIGHT
)
from .stereo_camera import StereoCameraRig, project_to_stereo


@dataclass
class FitResult:
    """Result of trajectory fitting."""
    params: PitchParams
    result: PitchResult
    residual: float
    success: bool
    n_iterations: int


def fit_trajectory_to_observations(
    observed_2d_wide: np.ndarray,
    observed_2d_ultra: np.ndarray,
    observed_depths: np.ndarray,
    timestamps: np.ndarray,
    rig: StereoCameraRig,
    initial_guess: Optional[dict] = None,
    method: str = 'L-BFGS-B'
) -> FitResult:
    """
    Fit pitch physics parameters to match stereo observations.
    
    This is the key insight from the research: instead of learning a direct
    mapping from 2D -> metrics, we optimize physics parameters that explain
    the observations. This is more robust and interpretable.
    
    Args:
        observed_2d_wide: (N, 2) pixel coords from wide camera
        observed_2d_ultra: (N, 2) pixel coords from ultrawide camera
        observed_depths: (N,) depth estimates (can have NaN)
        timestamps: (N,) time of each observation
        rig: Stereo camera configuration
        initial_guess: Optional dict with initial parameter estimates
        method: Optimization method
    
    Returns:
        FitResult with optimized parameters
    """
    # Filter valid observations
    valid = ~np.isnan(observed_2d_wide[:, 0])
    obs_wide = observed_2d_wide[valid]
    obs_ultra = observed_2d_ultra[valid]
    obs_depth = observed_depths[valid]
    obs_time = timestamps[valid]
    
    if len(obs_wide) < 5:
        # Not enough observations
        return FitResult(
            params=None, result=None, residual=float('inf'),
            success=False, n_iterations=0
        )
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            'velocity': 85.0,
            'spin_rate': 2200.0,
            'spin_axis_theta': 0.0,  # Spherical coords for spin axis
            'spin_axis_phi': np.pi / 2,
            'release_angle_h': 0.0,
            'release_angle_v': -2.0,
        }
    
    # Parameter bounds
    bounds = [
        (60, 105),      # velocity (mph)
        (500, 3500),    # spin_rate (rpm)
        (-np.pi, np.pi),  # spin_axis_theta
        (0, np.pi),     # spin_axis_phi
        (-10, 10),      # release_angle_h (degrees)
        (-10, 5),       # release_angle_v (degrees)
    ]
    
    # Initial parameter vector
    x0 = np.array([
        initial_guess['velocity'],
        initial_guess['spin_rate'],
        initial_guess.get('spin_axis_theta', 0.0),
        initial_guess.get('spin_axis_phi', np.pi / 2),
        initial_guess['release_angle_h'],
        initial_guess['release_angle_v'],
    ])
    
    # Objective function
    def objective(x):
        return _compute_residual(
            x, obs_wide, obs_ultra, obs_depth, obs_time, rig
        )
    
    # Optimize
    result = minimize(
        objective, x0, method=method, bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6}
    )
    
    # Extract final parameters
    final_params = _params_from_vector(result.x)
    final_result = simulate_pitch(final_params)
    
    return FitResult(
        params=final_params,
        result=final_result,
        residual=result.fun,
        success=result.success,
        n_iterations=result.nit
    )


def _compute_residual(
    x: np.ndarray,
    obs_wide: np.ndarray,
    obs_ultra: np.ndarray,
    obs_depth: np.ndarray,
    obs_time: np.ndarray,
    rig: StereoCameraRig
) -> float:
    """
    Compute reprojection + depth residual for given parameters.
    """
    params = _params_from_vector(x)
    
    try:
        result = simulate_pitch(params)
    except Exception:
        return 1e10  # Invalid parameters
    
    # Interpolate simulated trajectory at observation times
    sim_positions = np.zeros((len(obs_time), 3))
    for i, t in enumerate(obs_time):
        idx = np.searchsorted(result.times, t)
        if idx == 0:
            sim_positions[i] = result.positions[0]
        elif idx >= len(result.times):
            sim_positions[i] = result.positions[-1]
        else:
            # Linear interpolation
            t0, t1 = result.times[idx - 1], result.times[idx]
            alpha = (t - t0) / (t1 - t0)
            sim_positions[i] = (1 - alpha) * result.positions[idx - 1] + alpha * result.positions[idx]
    
    # Project simulated positions to cameras
    sim_wide, sim_ultra, sim_depth = project_to_stereo(sim_positions, rig)
    
    # Reprojection error (pixels)
    wide_error = np.nanmean((sim_wide - obs_wide) ** 2)
    ultra_error = np.nanmean((sim_ultra - obs_ultra) ** 2)
    
    # Depth error (meters) - weighted less since it's noisier
    depth_valid = ~np.isnan(obs_depth)
    if depth_valid.any():
        depth_error = np.mean((sim_depth[depth_valid] - obs_depth[depth_valid]) ** 2)
    else:
        depth_error = 0
    
    # Combined loss
    total = wide_error + ultra_error + 100 * depth_error
    
    return total


def _params_from_vector(x: np.ndarray) -> PitchParams:
    """Convert optimization vector to PitchParams."""
    velocity, spin_rate, theta, phi, angle_h, angle_v = x
    
    # Convert spherical to Cartesian for spin axis
    spin_axis = (
        np.sin(phi) * np.cos(theta),
        np.cos(phi),
        np.sin(phi) * np.sin(theta)
    )
    
    return PitchParams(
        velocity=velocity,
        spin_rate=spin_rate,
        spin_axis=spin_axis,
        release_angle_h=angle_h,
        release_angle_v=angle_v,
    )


class TrajectoryInitializer(nn.Module):
    """
    Neural network that predicts initial parameter estimates for the optimizer.
    This speeds up convergence by starting close to the solution.
    
    Input: 2D trajectory + depth observations
    Output: Initial guess for [velocity, spin_rate, spin_theta, spin_phi, angle_h, angle_v]
    """
    
    def __init__(self, hidden_dim: int = 128, max_seq_len: int = 50):
        super().__init__()
        
        # Input: (x_wide, y_wide, x_ultra, y_ultra, depth) per frame
        self.input_dim = 5
        self.output_dim = 6
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # Output scaling to reasonable parameter ranges
        self.register_buffer('output_scale', torch.tensor([
            20.0,   # velocity range ~20 mph
            1000.0, # spin range ~1000 rpm
            np.pi,  # theta range
            np.pi / 2,  # phi range
            5.0,    # angle_h range
            5.0,    # angle_v range
        ]))
        
        self.register_buffer('output_bias', torch.tensor([
            85.0,   # velocity center
            2200.0, # spin center
            0.0,    # theta center
            np.pi / 2,  # phi center
            0.0,    # angle_h center
            -2.0,   # angle_v center
        ]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 5) - [x_wide, y_wide, x_ultra, y_ultra, depth]
        
        Returns:
            (batch, 6) - initial parameter estimates
        """
        # Handle NaN values
        x = torch.nan_to_num(x, 0.0)
        
        # Normalize inputs
        x = self._normalize(x)
        
        # Encode
        h = self.encoder(x)
        
        # LSTM
        lstm_out, (h_n, _) = self.lstm(h)
        
        # Use final hidden states
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # Decode
        out = self.decoder(h_combined)
        
        # Scale to parameter ranges
        out = out * self.output_scale + self.output_bias
        
        return out
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input observations."""
        x = x.clone()
        # Pixel coords: normalize to [-1, 1]
        x[..., 0] = (x[..., 0] - 960) / 960
        x[..., 1] = (x[..., 1] - 540) / 540
        x[..., 2] = (x[..., 2] - 960) / 960
        x[..., 3] = (x[..., 3] - 540) / 540
        # Depth: normalize to ~[0, 1]
        x[..., 4] = x[..., 4] / 20.0
        return x


def fit_with_neural_init(
    observed_2d_wide: np.ndarray,
    observed_2d_ultra: np.ndarray,
    observed_depths: np.ndarray,
    timestamps: np.ndarray,
    rig: StereoCameraRig,
    initializer: TrajectoryInitializer,
    device: str = 'cuda'
) -> FitResult:
    """
    Fit trajectory using neural network for initialization.
    
    This combines the best of both worlds:
    1. Neural net provides fast, approximate initial guess
    2. Physics optimizer refines to exact solution
    """
    # Prepare input for neural net
    max_len = 50
    n = len(observed_2d_wide)
    
    # Stack observations
    obs = np.zeros((max_len, 5))
    obs[:n, 0] = observed_2d_wide[:, 0]
    obs[:n, 1] = observed_2d_wide[:, 1]
    obs[:n, 2] = observed_2d_ultra[:, 0]
    obs[:n, 3] = observed_2d_ultra[:, 1]
    obs[:n, 4] = observed_depths
    
    # Neural net inference
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        init_params = initializer(obs_tensor)[0].cpu().numpy()
    
    # Convert to initial guess dict
    initial_guess = {
        'velocity': float(init_params[0]),
        'spin_rate': float(init_params[1]),
        'spin_axis_theta': float(init_params[2]),
        'spin_axis_phi': float(init_params[3]),
        'release_angle_h': float(init_params[4]),
        'release_angle_v': float(init_params[5]),
    }
    
    # Run physics optimizer with neural init
    return fit_trajectory_to_observations(
        observed_2d_wide, observed_2d_ultra, observed_depths,
        timestamps, rig, initial_guess=initial_guess
    )
