"""
Complete video processing pipeline.
Takes video input → detects ball → tracks → outputs trajectory.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass
from pathlib import Path
import time

from .ball_detector import BallDetector, Detection
from .tracker import BallTracker, Track


@dataclass
class PipelineConfig:
    """Configuration for video pipeline."""
    # Detection
    detector_model: Optional[str] = None
    detection_confidence: float = 0.3
    
    # Tracking
    max_track_age: int = 10
    min_track_hits: int = 3
    max_association_distance: float = 100.0
    
    # Video
    target_fps: float = 60.0
    start_frame: int = 0
    end_frame: Optional[int] = None
    
    # Processing
    device: str = 'cuda'
    batch_size: int = 8


@dataclass
class PipelineResult:
    """Result from processing a video."""
    trajectory_2d: np.ndarray  # (N, 2) pixel coordinates
    timestamps: np.ndarray     # (N,) frame times
    confidences: np.ndarray    # (N,) detection confidences
    bboxes: List[Tuple[float, float, float, float]]
    
    # Metadata
    fps: float
    frame_count: int
    processing_time: float
    
    # Track info
    track_id: int
    track_length: int


class VideoPipeline:
    """
    End-to-end video processing pipeline.
    
    Usage:
        pipeline = VideoPipeline(config)
        result = pipeline.process_video("pitch.mp4")
        # result.trajectory_2d can be fed to physics fitter
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        self.detector = BallDetector(
            model_path=self.config.detector_model,
            confidence_threshold=self.config.detection_confidence,
            device=self.config.device
        )
        
        self.tracker = BallTracker(
            max_age=self.config.max_track_age,
            min_hits=self.config.min_track_hits,
            max_distance=self.config.max_association_distance
        )
    
    def process_video(self, video_path: str) -> Optional[PipelineResult]:
        """
        Process a video file and extract ball trajectory.
        
        Args:
            video_path: Path to video file
            
        Returns:
            PipelineResult with trajectory, or None if no ball found
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"  FPS: {fps}, Frames: {total_frames}")
        
        # Reset tracker
        self.tracker.reset()
        self.tracker.dt = 1.0 / fps
        
        start_time = time.time()
        frame_idx = 0
        
        # Skip to start frame
        if self.config.start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.start_frame)
            frame_idx = self.config.start_frame
        
        end_frame = self.config.end_frame or total_frames
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # Detect
            detections = self.detector.detect(frame)
            
            # Track
            self.tracker.update(detections, timestamp)
            
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{end_frame} frames")
        
        cap.release()
        processing_time = time.time() - start_time
        
        # Get best track
        best_track = self.tracker.get_best_track()
        
        if best_track is None or len(best_track.positions) < 5:
            print("  No valid ball trajectory found")
            return None
        
        print(f"  Found trajectory with {len(best_track.positions)} points")
        print(f"  Processing time: {processing_time:.2f}s")
        
        return PipelineResult(
            trajectory_2d=np.array(best_track.positions),
            timestamps=np.array(best_track.timestamps),
            confidences=np.array(best_track.confidences),
            bboxes=best_track.bboxes,
            fps=fps,
            frame_count=frame_idx,
            processing_time=processing_time,
            track_id=best_track.track_id,
            track_length=len(best_track.positions)
        )
    
    def process_frames(
        self,
        frames: List[np.ndarray],
        fps: float = 60.0
    ) -> Optional[PipelineResult]:
        """
        Process a list of frames (for stereo or pre-loaded video).
        
        Args:
            frames: List of BGR images
            fps: Frame rate
            
        Returns:
            PipelineResult with trajectory
        """
        self.tracker.reset()
        self.tracker.dt = 1.0 / fps
        
        start_time = time.time()
        
        for i, frame in enumerate(frames):
            timestamp = i / fps
            detections = self.detector.detect(frame)
            self.tracker.update(detections, timestamp)
        
        processing_time = time.time() - start_time
        
        best_track = self.tracker.get_best_track()
        
        if best_track is None or len(best_track.positions) < 5:
            return None
        
        return PipelineResult(
            trajectory_2d=np.array(best_track.positions),
            timestamps=np.array(best_track.timestamps),
            confidences=np.array(best_track.confidences),
            bboxes=best_track.bboxes,
            fps=fps,
            frame_count=len(frames),
            processing_time=processing_time,
            track_id=best_track.track_id,
            track_length=len(best_track.positions)
        )
    
    def process_stereo_video(
        self,
        wide_video_path: str,
        ultrawide_video_path: str
    ) -> Tuple[Optional[PipelineResult], Optional[PipelineResult]]:
        """
        Process synchronized stereo video pair.
        
        Args:
            wide_video_path: Path to wide camera video
            ultrawide_video_path: Path to ultrawide camera video
            
        Returns:
            Tuple of (wide_result, ultrawide_result)
        """
        # Process wide camera
        print("Processing wide camera...")
        wide_result = self.process_video(wide_video_path)
        
        # Reset and process ultrawide
        print("Processing ultrawide camera...")
        ultra_result = self.process_video(ultrawide_video_path)
        
        return wide_result, ultra_result


class RealtimePipeline:
    """
    Real-time processing pipeline for live camera feed.
    Yields results as frames are processed.
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        self.detector = BallDetector(
            model_path=self.config.detector_model,
            confidence_threshold=self.config.detection_confidence,
            device=self.config.device
        )
        
        self.tracker = BallTracker(
            max_age=self.config.max_track_age,
            min_hits=self.config.min_track_hits,
            max_distance=self.config.max_association_distance
        )
        
        self.frame_count = 0
        self.fps = self.config.target_fps
    
    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float = None
    ) -> Tuple[List[Detection], Optional[Track]]:
        """
        Process a single frame in real-time.
        
        Args:
            frame: BGR image
            timestamp: Frame timestamp (optional)
            
        Returns:
            Tuple of (detections, best_track)
        """
        if timestamp is None:
            timestamp = self.frame_count / self.fps
        
        # Detect
        detections = self.detector.detect(frame)
        
        # Track
        self.tracker.update(detections, timestamp)
        
        self.frame_count += 1
        
        return detections, self.tracker.get_best_track()
    
    def reset(self):
        """Reset pipeline state."""
        self.tracker.reset()
        self.frame_count = 0
    
    def get_trajectory(self) -> Optional[np.ndarray]:
        """Get current best trajectory."""
        track = self.tracker.get_best_track()
        if track:
            return track.to_trajectory_array()
        return None
