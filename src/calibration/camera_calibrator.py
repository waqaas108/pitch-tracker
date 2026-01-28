"""
Camera calibration from field markers.
Uses PnP (Perspective-n-Point) to estimate camera pose from known field geometry.
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .field_markers import FieldMarkers, STANDARD_FIELD_POINTS, get_calibration_point_options
from ..physics.stereo_camera import StereoCameraRig


@dataclass
class CalibrationResult:
    """Result of camera calibration."""
    # Camera pose
    position: np.ndarray      # (3,) world position
    rotation: np.ndarray      # (3, 3) rotation matrix
    
    # Intrinsics
    focal_length: float       # pixels
    principal_point: Tuple[float, float]
    
    # Quality
    reprojection_error: float
    num_points_used: int
    
    # For creating stereo rig
    def to_stereo_rig(self, image_size: Tuple[int, int] = (1920, 1080)) -> StereoCameraRig:
        """Convert calibration to stereo camera rig."""
        from ..physics.stereo_camera import StereoCameraRig
        
        # Approximate ultrawide position (8cm offset)
        baseline = 0.08
        right = self.rotation[:, 0]  # First column is right vector
        ultra_position = np.array([baseline, 0, 0])
        
        return StereoCameraRig(
            wide_position=np.zeros(3),
            ultrawide_position=ultra_position,
            wide_rotation=np.eye(3),
            ultrawide_rotation=np.eye(3),
            wide_focal=self.focal_length,
            ultrawide_focal=self.focal_length * 0.5,  # Ultrawide has wider FOV
            wide_principal=self.principal_point,
            ultrawide_principal=self.principal_point,
            image_width=image_size[0],
            image_height=image_size[1],
            rig_position=self.position,
            rig_rotation=self.rotation,
        )
    
    def save(self, path: str):
        """Save calibration to JSON."""
        data = {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'focal_length': self.focal_length,
            'principal_point': list(self.principal_point),
            'reprojection_error': self.reprojection_error,
            'num_points_used': self.num_points_used,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CalibrationResult':
        """Load calibration from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            position=np.array(data['position']),
            rotation=np.array(data['rotation']),
            focal_length=data['focal_length'],
            principal_point=tuple(data['principal_point']),
            reprojection_error=data['reprojection_error'],
            num_points_used=data['num_points_used'],
        )


class CameraCalibrator:
    """
    Calibrate camera from field marker correspondences.
    
    Usage:
        calibrator = CameraCalibrator()
        
        # Add point correspondences (pixel coords -> field point name)
        calibrator.add_point('home_plate_center', (960, 800))
        calibrator.add_point('pitching_rubber_center', (960, 400))
        calibrator.add_point('first_base', (1400, 600))
        calibrator.add_point('third_base', (520, 600))
        
        # Calibrate
        result = calibrator.calibrate()
        
        # Use result
        rig = result.to_stereo_rig()
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (1920, 1080),
        field_markers: FieldMarkers = None
    ):
        """
        Initialize calibrator.
        
        Args:
            image_size: (width, height) of video frames
            field_markers: Field geometry (default: standard MLB)
        """
        self.image_size = image_size
        self.field_markers = field_markers or STANDARD_FIELD_POINTS
        
        # Point correspondences
        self.correspondences: Dict[str, Tuple[float, float]] = {}
        
        # Initial focal length estimate (typical iPhone)
        self.initial_focal = 1400.0
    
    def add_point(self, point_name: str, pixel_coords: Tuple[float, float]):
        """
        Add a point correspondence.
        
        Args:
            point_name: Name of field marker (e.g., 'home_plate_center')
            pixel_coords: (x, y) pixel coordinates in image
        """
        # Validate point name
        world_point = self.field_markers.get_point(point_name)
        if world_point is None:
            valid = list(get_calibration_point_options().keys())
            raise ValueError(f"Unknown point '{point_name}'. Valid options: {valid}")
        
        self.correspondences[point_name] = pixel_coords
    
    def remove_point(self, point_name: str):
        """Remove a point correspondence."""
        if point_name in self.correspondences:
            del self.correspondences[point_name]
    
    def clear_points(self):
        """Clear all correspondences."""
        self.correspondences.clear()
    
    def calibrate(
        self,
        refine_focal: bool = True
    ) -> Optional[CalibrationResult]:
        """
        Perform camera calibration using PnP.
        
        Args:
            refine_focal: Whether to refine focal length estimate
            
        Returns:
            CalibrationResult or None if calibration failed
        """
        if len(self.correspondences) < 4:
            print(f"Need at least 4 points for calibration, have {len(self.correspondences)}")
            return None
        
        # Build arrays
        world_points = []
        image_points = []
        
        for name, pixel in self.correspondences.items():
            world_pt = self.field_markers.get_point(name)
            world_points.append(world_pt)
            image_points.append(pixel)
        
        world_points = np.array(world_points, dtype=np.float64)
        image_points = np.array(image_points, dtype=np.float64)
        
        # Camera matrix (initial estimate)
        cx, cy = self.image_size[0] / 2, self.image_size[1] / 2
        camera_matrix = np.array([
            [self.initial_focal, 0, cx],
            [0, self.initial_focal, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # No distortion (iPhone has minimal distortion)
        dist_coeffs = np.zeros(4)
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            world_points, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            print("PnP solver failed")
            return None
        
        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(
            world_points, image_points,
            camera_matrix, dist_coeffs,
            rvec, tvec
        )
        
        # Convert rotation vector to matrix
        rotation, _ = cv2.Rodrigues(rvec)
        
        # Camera position in world coordinates
        # tvec is the translation from world to camera
        # position = -R^T * t
        position = -rotation.T @ tvec.flatten()
        
        # Compute reprojection error
        projected, _ = cv2.projectPoints(
            world_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        projected = projected.reshape(-1, 2)
        
        errors = np.sqrt(np.sum((projected - image_points) ** 2, axis=1))
        mean_error = np.mean(errors)
        
        # Optionally refine focal length
        focal = self.initial_focal
        if refine_focal and len(self.correspondences) >= 6:
            focal = self._refine_focal_length(
                world_points, image_points, rotation, position
            )
        
        return CalibrationResult(
            position=position,
            rotation=rotation,
            focal_length=focal,
            principal_point=(cx, cy),
            reprojection_error=mean_error,
            num_points_used=len(self.correspondences)
        )
    
    def _refine_focal_length(
        self,
        world_points: np.ndarray,
        image_points: np.ndarray,
        rotation: np.ndarray,
        position: np.ndarray
    ) -> float:
        """Refine focal length estimate using least squares."""
        from scipy.optimize import minimize_scalar
        
        cx, cy = self.image_size[0] / 2, self.image_size[1] / 2
        
        def error_fn(focal):
            camera_matrix = np.array([
                [focal, 0, cx],
                [0, focal, cy],
                [0, 0, 1]
            ])
            
            # Project points
            tvec = -rotation @ position
            rvec, _ = cv2.Rodrigues(rotation)
            
            projected, _ = cv2.projectPoints(
                world_points, rvec, tvec, camera_matrix, np.zeros(4)
            )
            projected = projected.reshape(-1, 2)
            
            return np.mean(np.sum((projected - image_points) ** 2, axis=1))
        
        result = minimize_scalar(error_fn, bounds=(500, 3000), method='bounded')
        
        return result.x if result.success else self.initial_focal
    
    def calibrate_interactive(self, frame: np.ndarray) -> Optional[CalibrationResult]:
        """
        Interactive calibration using OpenCV window.
        Click on field markers to add correspondences.
        
        Args:
            frame: Video frame to calibrate on
            
        Returns:
            CalibrationResult after user completes calibration
        """
        print("\n" + "="*50)
        print("INTERACTIVE CAMERA CALIBRATION")
        print("="*50)
        print("\nInstructions:")
        print("1. Click on visible field markers in the image")
        print("2. Enter the marker name when prompted")
        print("3. Add at least 4 points (6+ recommended)")
        print("4. Press 'c' to calibrate, 'q' to quit")
        print("\nAvailable markers:")
        for name, display in get_calibration_point_options().items():
            print(f"  {name}: {display}")
        
        self.clear_points()
        click_point = [None]
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_point[0] = (x, y)
        
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', mouse_callback)
        
        display_frame = frame.copy()
        
        while True:
            # Draw existing points
            temp_frame = display_frame.copy()
            for name, pixel in self.correspondences.items():
                cv2.circle(temp_frame, (int(pixel[0]), int(pixel[1])), 5, (0, 255, 0), -1)
                cv2.putText(temp_frame, name[:10], (int(pixel[0])+10, int(pixel[1])),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show point count
            cv2.putText(temp_frame, f"Points: {len(self.correspondences)}/4+",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', temp_frame)
            key = cv2.waitKey(100) & 0xFF
            
            # Handle click
            if click_point[0] is not None:
                x, y = click_point[0]
                click_point[0] = None
                
                # Prompt for marker name
                print(f"\nClicked at ({x}, {y})")
                name = input("Enter marker name (or 'skip'): ").strip()
                
                if name and name != 'skip':
                    try:
                        self.add_point(name, (x, y))
                        print(f"Added {name}")
                    except ValueError as e:
                        print(f"Error: {e}")
            
            # Handle keys
            if key == ord('c'):
                if len(self.correspondences) >= 4:
                    result = self.calibrate()
                    if result:
                        print(f"\nCalibration successful!")
                        print(f"Camera position: {result.position}")
                        print(f"Reprojection error: {result.reprojection_error:.2f} pixels")
                        cv2.destroyAllWindows()
                        return result
                    else:
                        print("Calibration failed, add more points")
                else:
                    print(f"Need at least 4 points, have {len(self.correspondences)}")
            
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return None


def calibrate_from_file(
    image_path: str,
    output_path: str = None
) -> Optional[CalibrationResult]:
    """
    Convenience function to calibrate from an image file.
    
    Args:
        image_path: Path to image/video frame
        output_path: Optional path to save calibration JSON
        
    Returns:
        CalibrationResult
    """
    import cv2
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return None
    
    calibrator = CameraCalibrator(image_size=(frame.shape[1], frame.shape[0]))
    result = calibrator.calibrate_interactive(frame)
    
    if result and output_path:
        result.save(output_path)
        print(f"Saved calibration to {output_path}")
    
    return result
