"""
Camera projection utilities for converting 3D trajectories to 2D image coordinates.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters."""
    # Intrinsics (typical iPhone 15 Pro main camera approximation)
    focal_length: float = 26.0  # mm (35mm equivalent)
    sensor_width: float = 6.86  # mm (approximate)
    sensor_height: float = 5.14  # mm
    image_width: int = 1920
    image_height: int = 1080
    
    # Extrinsics - camera position in world coords (meters)
    position: Tuple[float, float, float] = (-3.0, 1.5, -2.0)  # behind and to side of mound
    
    # Camera orientation (look-at point)
    look_at: Tuple[float, float, float] = (0.0, 1.0, 9.0)  # roughly at plate
    
    @property
    def fx(self) -> float:
        """Focal length in pixels (x)."""
        return self.focal_length * self.image_width / self.sensor_width
    
    @property
    def fy(self) -> float:
        """Focal length in pixels (y)."""
        return self.focal_length * self.image_height / self.sensor_height
    
    @property
    def cx(self) -> float:
        """Principal point x."""
        return self.image_width / 2
    
    @property
    def cy(self) -> float:
        """Principal point y."""
        return self.image_height / 2


def get_camera_matrix(params: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera intrinsic matrix K and extrinsic matrix [R|t].
    
    Returns:
        K: 3x3 intrinsic matrix
        Rt: 3x4 extrinsic matrix [R|t]
    """
    # Intrinsic matrix
    K = np.array([
        [params.fx, 0, params.cx],
        [0, params.fy, params.cy],
        [0, 0, 1]
    ])
    
    # Extrinsic: camera pose
    cam_pos = np.array(params.position)
    look_at = np.array(params.look_at)
    
    # Camera coordinate system
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    
    up_world = np.array([0, 1, 0])
    right = np.cross(forward, up_world)
    right = right / np.linalg.norm(right)
    
    up = np.cross(right, forward)
    
    # Rotation matrix (world to camera)
    R = np.array([right, -up, forward])
    
    # Translation
    t = -R @ cam_pos
    
    Rt = np.hstack([R, t.reshape(3, 1)])
    
    return K, Rt


def project_points(points_3d: np.ndarray, camera: CameraParams) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        camera: Camera parameters
        
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
    """
    K, Rt = get_camera_matrix(camera)
    
    # Homogeneous coordinates
    N = points_3d.shape[0]
    points_h = np.hstack([points_3d, np.ones((N, 1))])
    
    # Project: p = K @ Rt @ P
    P = K @ Rt
    projected = (P @ points_h.T).T
    
    # Normalize
    points_2d = projected[:, :2] / projected[:, 2:3]
    
    return points_2d


def add_detection_noise(points_2d: np.ndarray, 
                        noise_std: float = 2.0,
                        dropout_prob: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add realistic detection noise to 2D points.
    
    Args:
        points_2d: (N, 2) clean 2D points
        noise_std: standard deviation of Gaussian noise in pixels
        dropout_prob: probability of missing detection per frame
        
    Returns:
        noisy_points: (N, 2) noisy points (NaN for dropouts)
        valid_mask: (N,) boolean mask of valid detections
    """
    N = points_2d.shape[0]
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, points_2d.shape)
    noisy = points_2d + noise
    
    # Random dropouts
    valid_mask = np.random.random(N) > dropout_prob
    noisy[~valid_mask] = np.nan
    
    return noisy, valid_mask


def get_camera_presets() -> dict:
    """Return common camera position presets."""
    return {
        'behind_mound_left': CameraParams(
            position=(-3.0, 1.8, -1.0),
            look_at=(0.0, 1.0, 10.0)
        ),
        'behind_mound_right': CameraParams(
            position=(3.0, 1.8, -1.0),
            look_at=(0.0, 1.0, 10.0)
        ),
        'behind_mound_center': CameraParams(
            position=(0.0, 2.5, -2.0),
            look_at=(0.0, 0.8, 10.0)
        ),
        'side_first_base': CameraParams(
            position=(12.0, 1.5, 9.0),
            look_at=(0.0, 1.0, 9.0)
        ),
        'side_third_base': CameraParams(
            position=(-12.0, 1.5, 9.0),
            look_at=(0.0, 1.0, 9.0)
        ),
        'high_home': CameraParams(
            position=(0.0, 4.0, 20.0),
            look_at=(0.0, 1.0, 9.0)
        ),
    }
