"""
Stereo camera simulation for iPhone-like dual camera setup.
Models wide + ultrawide cameras with realistic baseline and intrinsics.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
import torch


@dataclass
class StereoCameraRig:
    """
    iPhone-like stereo camera configuration.
    Based on iPhone 15 Pro specs: ~8cm baseline between wide and ultrawide.
    """
    # Camera positions (meters) - relative to rig center
    wide_position: np.ndarray  # Main camera
    ultrawide_position: np.ndarray  # Secondary camera
    
    # Camera orientations (rotation matrices)
    wide_rotation: np.ndarray
    ultrawide_rotation: np.ndarray
    
    # Intrinsics (focal length in pixels, principal point)
    wide_focal: float
    ultrawide_focal: float
    wide_principal: Tuple[float, float]
    ultrawide_principal: Tuple[float, float]
    
    # Image dimensions
    image_width: int = 1920
    image_height: int = 1080
    
    # Rig pose in world coordinates
    rig_position: np.ndarray = None
    rig_rotation: np.ndarray = None
    
    def __post_init__(self):
        if self.rig_position is None:
            self.rig_position = np.zeros(3)
        if self.rig_rotation is None:
            self.rig_rotation = np.eye(3)


def create_iphone_stereo_rig(
    position: np.ndarray,
    look_at: np.ndarray,
    up: np.ndarray = None
) -> StereoCameraRig:
    """
    Create an iPhone-like stereo camera rig.
    
    Args:
        position: World position of the phone
        look_at: Point the phone is looking at
        up: Up vector (default: world Y-up)
    
    Returns:
        StereoCameraRig configured like iPhone 15 Pro
    """
    if up is None:
        up = np.array([0, 1, 0])
    
    # Compute rig orientation (look-at matrix)
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    # Rotation matrix: columns are right, up, -forward (OpenGL convention)
    rig_rotation = np.column_stack([right, true_up, -forward])
    
    # iPhone camera layout (approximate):
    # Wide camera is roughly centered
    # Ultrawide is offset ~8cm to the side
    baseline = 0.08  # 8cm baseline
    
    wide_position = np.array([0, 0, 0])
    ultrawide_position = np.array([baseline, 0, 0])  # Offset in local X
    
    # Focal lengths (pixels) for 1920x1080
    # Wide: ~26mm equivalent, ultrawide: ~13mm equivalent
    # Sensor crop factor ~7x for iPhone
    wide_focal = 1400  # Approximate for 26mm on 1080p
    ultrawide_focal = 700  # Approximate for 13mm on 1080p
    
    principal = (960, 540)  # Image center
    
    return StereoCameraRig(
        wide_position=wide_position,
        ultrawide_position=ultrawide_position,
        wide_rotation=np.eye(3),  # Both cameras look forward
        ultrawide_rotation=np.eye(3),
        wide_focal=wide_focal,
        ultrawide_focal=ultrawide_focal,
        wide_principal=principal,
        ultrawide_principal=principal,
        rig_position=position,
        rig_rotation=rig_rotation,
    )


def project_to_stereo(
    points_3d: np.ndarray,
    rig: StereoCameraRig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D points to both cameras in the stereo rig.
    
    Args:
        points_3d: (N, 3) world coordinates
        rig: Stereo camera configuration
    
    Returns:
        wide_2d: (N, 2) pixel coordinates in wide camera
        ultrawide_2d: (N, 2) pixel coordinates in ultrawide camera  
        depths: (N,) depth values from wide camera
    """
    # Transform to rig coordinates
    points_rig = (points_3d - rig.rig_position) @ rig.rig_rotation
    
    # Project to wide camera
    points_wide = points_rig - rig.wide_position
    points_wide = points_wide @ rig.wide_rotation
    
    wide_2d = np.zeros((len(points_3d), 2))
    wide_2d[:, 0] = rig.wide_focal * points_wide[:, 0] / points_wide[:, 2] + rig.wide_principal[0]
    wide_2d[:, 1] = rig.wide_focal * points_wide[:, 1] / points_wide[:, 2] + rig.wide_principal[1]
    
    # Project to ultrawide camera
    points_ultra = points_rig - rig.ultrawide_position
    points_ultra = points_ultra @ rig.ultrawide_rotation
    
    ultrawide_2d = np.zeros((len(points_3d), 2))
    ultrawide_2d[:, 0] = rig.ultrawide_focal * points_ultra[:, 0] / points_ultra[:, 2] + rig.ultrawide_principal[0]
    ultrawide_2d[:, 1] = rig.ultrawide_focal * points_ultra[:, 1] / points_ultra[:, 2] + rig.ultrawide_principal[1]
    
    # Depth from wide camera
    depths = points_wide[:, 2]
    
    return wide_2d, ultrawide_2d, depths


def triangulate_stereo(
    wide_2d: np.ndarray,
    ultrawide_2d: np.ndarray,
    rig: StereoCameraRig
) -> np.ndarray:
    """
    Triangulate 3D points from stereo correspondences.
    Uses linear triangulation (DLT method).
    
    Args:
        wide_2d: (N, 2) pixel coordinates in wide camera
        ultrawide_2d: (N, 2) pixel coordinates in ultrawide camera
        rig: Stereo camera configuration
    
    Returns:
        points_3d: (N, 3) world coordinates
    """
    n_points = len(wide_2d)
    points_3d = np.zeros((n_points, 3))
    
    for i in range(n_points):
        # Build projection matrices
        P1 = _build_projection_matrix(
            rig.wide_focal, rig.wide_principal,
            rig.wide_rotation, rig.wide_position,
            rig.rig_rotation, rig.rig_position
        )
        P2 = _build_projection_matrix(
            rig.ultrawide_focal, rig.ultrawide_principal,
            rig.ultrawide_rotation, rig.ultrawide_position,
            rig.rig_rotation, rig.rig_position
        )
        
        # DLT triangulation
        A = np.zeros((4, 4))
        A[0] = wide_2d[i, 0] * P1[2] - P1[0]
        A[1] = wide_2d[i, 1] * P1[2] - P1[1]
        A[2] = ultrawide_2d[i, 0] * P2[2] - P2[0]
        A[3] = ultrawide_2d[i, 1] * P2[2] - P2[1]
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_3d[i] = X[:3] / X[3]
    
    return points_3d


def _build_projection_matrix(
    focal: float,
    principal: Tuple[float, float],
    cam_rotation: np.ndarray,
    cam_position: np.ndarray,
    rig_rotation: np.ndarray,
    rig_position: np.ndarray
) -> np.ndarray:
    """Build 3x4 projection matrix for a camera."""
    # Intrinsic matrix
    K = np.array([
        [focal, 0, principal[0]],
        [0, focal, principal[1]],
        [0, 0, 1]
    ])
    
    # Extrinsic: world to camera
    R_cam = cam_rotation @ rig_rotation.T
    t_cam = -R_cam @ (rig_position + rig_rotation @ cam_position)
    
    # Projection matrix
    Rt = np.column_stack([R_cam, t_cam])
    P = K @ Rt
    
    return P


def add_stereo_noise(
    wide_2d: np.ndarray,
    ultrawide_2d: np.ndarray,
    depths: np.ndarray,
    pixel_noise_std: float = 2.0,
    depth_noise_std: float = 0.1,
    dropout_prob: float = 0.05,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Add realistic noise to stereo observations.
    
    Returns:
        noisy_wide, noisy_ultra, noisy_depths, valid_mask
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(wide_2d)
    
    # Pixel noise
    noisy_wide = wide_2d + rng.normal(0, pixel_noise_std, wide_2d.shape)
    noisy_ultra = ultrawide_2d + rng.normal(0, pixel_noise_std * 1.5, ultrawide_2d.shape)  # Ultrawide slightly noisier
    
    # Depth noise (proportional to distance)
    depth_noise = rng.normal(0, depth_noise_std, depths.shape) * (depths / 10)
    noisy_depths = depths + depth_noise
    
    # Random dropouts
    valid_mask = rng.random(n) > dropout_prob
    
    # Mark invalid points with NaN
    noisy_wide[~valid_mask] = np.nan
    noisy_ultra[~valid_mask] = np.nan
    noisy_depths[~valid_mask] = np.nan
    
    return noisy_wide, noisy_ultra, noisy_depths, valid_mask


def get_stereo_camera_presets() -> dict:
    """
    Get preset stereo camera positions for common filming angles.
    """
    # Mound-to-plate distance
    mound_z = 0
    plate_z = 18.44
    
    presets = {}
    
    # Behind and above the mound (classic broadcast angle)
    presets['behind_high'] = create_iphone_stereo_rig(
        position=np.array([0, 3.0, -2.0]),
        look_at=np.array([0, 1.0, plate_z / 2])
    )
    
    # Behind mound, lower angle
    presets['behind_low'] = create_iphone_stereo_rig(
        position=np.array([0, 1.8, -1.5]),
        look_at=np.array([0, 0.8, plate_z / 2])
    )
    
    # Side view (first base side)
    presets['side_1b'] = create_iphone_stereo_rig(
        position=np.array([12, 2.0, plate_z / 2]),
        look_at=np.array([0, 1.2, plate_z / 2])
    )
    
    # Side view (third base side)
    presets['side_3b'] = create_iphone_stereo_rig(
        position=np.array([-12, 2.0, plate_z / 2]),
        look_at=np.array([0, 1.2, plate_z / 2])
    )
    
    # Behind catcher (umpire view)
    presets['behind_catcher'] = create_iphone_stereo_rig(
        position=np.array([0, 1.5, plate_z + 2]),
        look_at=np.array([0, 1.5, 0])
    )
    
    return presets
