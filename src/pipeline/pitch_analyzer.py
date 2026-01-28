"""
Complete pitch analysis pipeline.
Video → Detection → Tracking → Physics Fitting → Metrics
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from pathlib import Path
import torch
import json

from ..detection.video_pipeline import VideoPipeline, PipelineConfig, PipelineResult
from ..physics.stereo_camera import StereoCameraRig, create_iphone_stereo_rig, triangulate_stereo
from ..physics.trajectory_fitter import (
    fit_trajectory_to_observations,
    fit_with_neural_init,
    TrajectoryInitializer,
    FitResult
)
from ..data.pitch_types import PITCH_PROFILES, get_label_to_pitch_type
from ..models.pitch_classifier import PitchClassifier


@dataclass
class PitchAnalysisResult:
    """Complete analysis result for a single pitch."""
    # Core metrics
    velocity: float           # mph
    horizontal_break: float   # inches
    vertical_break: float     # inches
    
    # Spin (estimated)
    spin_rate: float          # rpm
    spin_axis: Tuple[float, float, float]
    
    # Location
    plate_location: Tuple[float, float]  # (x, y) in meters
    
    # Classification
    pitch_type: str           # e.g., 'FF', 'CU', 'SL'
    pitch_type_name: str      # e.g., 'Four-Seam Fastball'
    pitch_type_confidence: float
    
    # Quality metrics
    fit_residual: float
    trajectory_length: int
    
    # Raw data
    trajectory_2d: np.ndarray
    timestamps: np.ndarray
    
    def to_dict(self) -> dict:
        return {
            'velocity_mph': round(self.velocity, 1),
            'horizontal_break_inches': round(self.horizontal_break, 1),
            'vertical_break_inches': round(self.vertical_break, 1),
            'spin_rate_rpm': round(self.spin_rate),
            'spin_axis': [round(x, 3) for x in self.spin_axis],
            'plate_location': [round(x, 3) for x in self.plate_location],
            'pitch_type': self.pitch_type,
            'pitch_type_name': self.pitch_type_name,
            'pitch_type_confidence': round(self.pitch_type_confidence, 2),
            'fit_residual': round(self.fit_residual, 2),
            'trajectory_points': self.trajectory_length,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.pitch_type_name}\n"
            f"  Velocity: {self.velocity:.1f} mph\n"
            f"  H-Break: {self.horizontal_break:.1f}\"\n"
            f"  V-Break: {self.vertical_break:.1f}\"\n"
            f"  Spin: {self.spin_rate:.0f} rpm\n"
            f"  Location: ({self.plate_location[0]:.2f}, {self.plate_location[1]:.2f})"
        )


class PitchAnalyzer:
    """
    End-to-end pitch analysis from video.
    
    Supports:
    - Single camera (monocular) - less accurate
    - Stereo cameras (wide + ultrawide) - more accurate
    - Pre-extracted trajectories
    
    Usage:
        analyzer = PitchAnalyzer()
        analyzer.load_models("checkpoints/stereo")
        
        # From video
        result = analyzer.analyze_video("pitch.mp4")
        
        # From stereo video pair
        result = analyzer.analyze_stereo_video("wide.mp4", "ultra.mp4")
        
        print(result)
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        camera_position: np.ndarray = None,
        camera_look_at: np.ndarray = None
    ):
        """
        Initialize analyzer.
        
        Args:
            device: 'cuda' or 'cpu'
            camera_position: World position of camera (meters)
            camera_look_at: Point camera is looking at
        """
        self.device = device
        
        # Default camera setup: behind mound, elevated
        if camera_position is None:
            camera_position = np.array([0, 2.5, -2.0])
        if camera_look_at is None:
            camera_look_at = np.array([0, 1.0, 9.0])
        
        self.camera_rig = create_iphone_stereo_rig(camera_position, camera_look_at)
        
        # Models (loaded lazily)
        self.initializer: Optional[TrajectoryInitializer] = None
        self.classifier: Optional[PitchClassifier] = None
        
        # Video pipeline
        self.pipeline = VideoPipeline(PipelineConfig(device=device))
    
    def load_models(self, model_dir: str):
        """Load trained models."""
        model_path = Path(model_dir)
        
        # Load initializer
        init_path = model_path / 'best_initializer.pt'
        if init_path.exists():
            self.initializer = TrajectoryInitializer().to(self.device)
            ckpt = torch.load(init_path, map_location=self.device, weights_only=True)
            self.initializer.load_state_dict(ckpt['model_state_dict'])
            self.initializer.eval()
            print(f"Loaded initializer from {init_path}")
        
        # Load classifier (optional)
        class_path = model_path.parent / 'classifier' / 'best_model.pt'
        if class_path.exists():
            self.classifier = PitchClassifier(num_classes=len(PITCH_PROFILES)).to(self.device)
            ckpt = torch.load(class_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(ckpt['model_state_dict'])
            self.classifier.eval()
            print(f"Loaded classifier from {class_path}")
    
    def analyze_video(
        self,
        video_path: str,
        estimate_depth: bool = True
    ) -> Optional[PitchAnalysisResult]:
        """
        Analyze a single-camera video.
        
        Args:
            video_path: Path to video file
            estimate_depth: Whether to estimate depth from motion
            
        Returns:
            PitchAnalysisResult or None if analysis failed
        """
        # Extract trajectory
        pipeline_result = self.pipeline.process_video(video_path)
        
        if pipeline_result is None:
            return None
        
        # For monocular, we need to estimate depth
        # Use trajectory motion to infer approximate depth
        if estimate_depth:
            depths = self._estimate_depth_from_motion(
                pipeline_result.trajectory_2d,
                pipeline_result.timestamps
            )
        else:
            depths = np.full(len(pipeline_result.trajectory_2d), np.nan)
        
        # Create dummy ultrawide (same as wide for monocular)
        ultra_2d = pipeline_result.trajectory_2d.copy()
        
        return self._fit_and_classify(
            pipeline_result.trajectory_2d,
            ultra_2d,
            depths,
            pipeline_result.timestamps
        )
    
    def analyze_stereo_video(
        self,
        wide_video_path: str,
        ultrawide_video_path: str
    ) -> Optional[PitchAnalysisResult]:
        """
        Analyze synchronized stereo video pair.
        
        Args:
            wide_video_path: Path to wide camera video
            ultrawide_video_path: Path to ultrawide camera video
            
        Returns:
            PitchAnalysisResult or None
        """
        wide_result, ultra_result = self.pipeline.process_stereo_video(
            wide_video_path, ultrawide_video_path
        )
        
        if wide_result is None or ultra_result is None:
            return None
        
        # Align trajectories by timestamp
        wide_2d, ultra_2d, timestamps = self._align_trajectories(
            wide_result, ultra_result
        )
        
        # Triangulate depth
        depths = self._triangulate_depths(wide_2d, ultra_2d)
        
        return self._fit_and_classify(wide_2d, ultra_2d, depths, timestamps)
    
    def analyze_trajectory(
        self,
        wide_2d: np.ndarray,
        ultrawide_2d: np.ndarray,
        depths: np.ndarray,
        timestamps: np.ndarray
    ) -> Optional[PitchAnalysisResult]:
        """
        Analyze pre-extracted trajectory data.
        
        Args:
            wide_2d: (N, 2) pixel coords from wide camera
            ultrawide_2d: (N, 2) pixel coords from ultrawide camera
            depths: (N,) depth estimates
            timestamps: (N,) frame times
            
        Returns:
            PitchAnalysisResult or None
        """
        return self._fit_and_classify(wide_2d, ultrawide_2d, depths, timestamps)
    
    def _fit_and_classify(
        self,
        wide_2d: np.ndarray,
        ultra_2d: np.ndarray,
        depths: np.ndarray,
        timestamps: np.ndarray
    ) -> Optional[PitchAnalysisResult]:
        """Core fitting and classification logic."""
        
        # Fit physics parameters
        if self.initializer is not None:
            fit_result = fit_with_neural_init(
                wide_2d, ultra_2d, depths, timestamps,
                self.camera_rig, self.initializer, self.device
            )
        else:
            fit_result = fit_trajectory_to_observations(
                wide_2d, ultra_2d, depths, timestamps, self.camera_rig
            )
        
        if not fit_result.success or fit_result.params is None:
            return None
        
        # Classify pitch type
        pitch_type, confidence = self._classify_pitch(
            wide_2d, fit_result
        )
        
        return PitchAnalysisResult(
            velocity=fit_result.params.velocity,
            horizontal_break=fit_result.result.horizontal_break,
            vertical_break=fit_result.result.vertical_break,
            spin_rate=fit_result.params.spin_rate,
            spin_axis=fit_result.params.spin_axis,
            plate_location=fit_result.result.plate_location,
            pitch_type=pitch_type,
            pitch_type_name=PITCH_PROFILES[pitch_type].name,
            pitch_type_confidence=confidence,
            fit_residual=fit_result.residual,
            trajectory_length=len(wide_2d),
            trajectory_2d=wide_2d,
            timestamps=timestamps
        )
    
    def _classify_pitch(
        self,
        trajectory_2d: np.ndarray,
        fit_result: FitResult
    ) -> Tuple[str, float]:
        """Classify pitch type from trajectory and fitted metrics."""
        
        if self.classifier is not None:
            # Use trained classifier
            # Prepare input
            max_len = 50
            traj = np.zeros((max_len, 2))
            n = min(len(trajectory_2d), max_len)
            traj[:n] = trajectory_2d[:n]
            
            traj_tensor = torch.tensor(traj, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            metrics = torch.tensor([
                fit_result.params.velocity,
                fit_result.result.horizontal_break,
                fit_result.result.vertical_break,
                fit_result.result.plate_location[0],
                fit_result.result.plate_location[1],
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.classifier(trajectory=traj_tensor, metrics=metrics)
                probs = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            label_to_type = get_label_to_pitch_type()
            pitch_type = label_to_type[pred_class]
            
        else:
            # Rule-based classification fallback
            pitch_type, confidence = self._rule_based_classify(fit_result)
        
        return pitch_type, confidence
    
    def _rule_based_classify(self, fit_result: FitResult) -> Tuple[str, float]:
        """Simple rule-based pitch classification."""
        v = fit_result.params.velocity
        h_break = fit_result.result.horizontal_break
        v_break = fit_result.result.vertical_break
        
        # Simple decision tree based on velocity and break
        if v >= 92:
            if abs(h_break) < 3 and v_break < 0:
                return 'FF', 0.7  # Four-seam fastball
            elif h_break < -2:
                return 'SI', 0.6  # Sinker
            else:
                return 'FC', 0.5  # Cutter
        elif v >= 82:
            if h_break < -3:
                return 'SL', 0.6  # Slider
            elif v_break > 2:
                return 'CU', 0.5  # Curveball
            else:
                return 'CH', 0.5  # Changeup
        else:
            if v_break > 3:
                return 'CU', 0.7  # Curveball
            else:
                return 'CH', 0.5  # Changeup
    
    def _estimate_depth_from_motion(
        self,
        trajectory_2d: np.ndarray,
        timestamps: np.ndarray
    ) -> np.ndarray:
        """
        Estimate depth from 2D motion (monocular depth cue).
        
        Objects moving toward camera appear to accelerate in image space.
        This is a rough approximation.
        """
        n = len(trajectory_2d)
        depths = np.zeros(n)
        
        # Compute apparent velocity in pixels
        for i in range(1, n):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = trajectory_2d[i, 0] - trajectory_2d[i-1, 0]
                dy = trajectory_2d[i, 1] - trajectory_2d[i-1, 1]
                pixel_vel = np.sqrt(dx**2 + dy**2) / dt
                
                # Rough depth estimate: faster apparent motion = closer
                # Assume ball travels ~40 m/s, focal length ~1400 pixels
                # depth ≈ focal * real_vel / pixel_vel
                if pixel_vel > 10:
                    depths[i] = 1400 * 40 / pixel_vel
                else:
                    depths[i] = 20  # Far away default
        
        depths[0] = depths[1] if n > 1 else 18  # Mound distance
        
        # Smooth depths
        from scipy.ndimage import uniform_filter1d
        depths = uniform_filter1d(depths, size=3)
        
        return depths
    
    def _align_trajectories(
        self,
        wide_result: PipelineResult,
        ultra_result: PipelineResult
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align two trajectories by timestamp."""
        # Find common timestamps (within tolerance)
        tolerance = 0.02  # 20ms
        
        wide_times = wide_result.timestamps
        ultra_times = ultra_result.timestamps
        
        aligned_wide = []
        aligned_ultra = []
        aligned_times = []
        
        for i, t_w in enumerate(wide_times):
            # Find closest ultrawide timestamp
            diffs = np.abs(ultra_times - t_w)
            j = np.argmin(diffs)
            
            if diffs[j] < tolerance:
                aligned_wide.append(wide_result.trajectory_2d[i])
                aligned_ultra.append(ultra_result.trajectory_2d[j])
                aligned_times.append(t_w)
        
        return (
            np.array(aligned_wide),
            np.array(aligned_ultra),
            np.array(aligned_times)
        )
    
    def _triangulate_depths(
        self,
        wide_2d: np.ndarray,
        ultra_2d: np.ndarray
    ) -> np.ndarray:
        """Triangulate depth from stereo correspondences."""
        points_3d = triangulate_stereo(wide_2d, ultra_2d, self.camera_rig)
        
        # Extract depth (Z coordinate in camera frame)
        # Transform to camera coordinates
        points_cam = (points_3d - self.camera_rig.rig_position) @ self.camera_rig.rig_rotation
        depths = points_cam[:, 2]
        
        return depths
