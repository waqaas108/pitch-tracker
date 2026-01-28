"""
Ball tracking across frames.
Links detections into continuous trajectories using Kalman filtering + association.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque

from .ball_detector import Detection


@dataclass
class Track:
    """A tracked ball trajectory."""
    track_id: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    
    # Kalman filter state
    state: np.ndarray = None  # [x, y, vx, vy]
    covariance: np.ndarray = None
    
    # Track status
    age: int = 0
    hits: int = 0
    misses: int = 0
    
    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed after enough hits."""
        return self.hits >= 3
    
    @property
    def is_dead(self) -> bool:
        """Track is dead after too many misses."""
        return self.misses > 5
    
    def to_trajectory_array(self) -> np.ndarray:
        """Convert to (N, 2) array for physics fitting."""
        return np.array(self.positions)


class BallTracker:
    """
    Multi-object tracker for baseball detection.
    
    Uses Kalman filter for motion prediction and Hungarian algorithm
    for detection-to-track association. Optimized for single ball tracking
    but handles multiple candidates.
    """
    
    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 100.0
    ):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections to confirm track
            iou_threshold: IoU threshold for association
            max_distance: Maximum pixel distance for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        
        self.tracks: List[Track] = []
        self.next_id = 0
        self.frame_count = 0
        
        # Kalman filter parameters
        self.dt = 1.0 / 60.0  # Assume 60fps
        self._init_kalman_matrices()
    
    def _init_kalman_matrices(self):
        """Initialize Kalman filter matrices."""
        # State transition: constant velocity model
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix: we observe position only
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise
        q = 100.0  # Position variance
        self.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*10, 0],
            [0, 0, 0, q*10]
        ])
        
        # Measurement noise
        self.R = np.array([
            [25.0, 0],
            [0, 25.0]
        ])
    
    def update(
        self,
        detections: List[Detection],
        timestamp: float = None
    ) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections in current frame
            timestamp: Frame timestamp (optional)
            
        Returns:
            List of active tracks
        """
        if timestamp is None:
            timestamp = self.frame_count * self.dt
        
        self.frame_count += 1
        
        # Predict existing tracks
        for track in self.tracks:
            self._predict(track)
        
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._associate(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            self._update_track(self.tracks[track_idx], detections[det_idx], timestamp)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].misses += 1
            self.tracks[track_idx].age += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], timestamp)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_dead]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed]
    
    def _predict(self, track: Track):
        """Kalman predict step."""
        if track.state is None:
            return
        
        track.state = self.F @ track.state
        track.covariance = self.F @ track.covariance @ self.F.T + self.Q
    
    def _update_track(self, track: Track, detection: Detection, timestamp: float):
        """Kalman update step."""
        z = np.array([detection.center[0], detection.center[1]])
        
        if track.state is None:
            # Initialize state from first detection
            track.state = np.array([z[0], z[1], 0, 0])
            track.covariance = np.eye(4) * 100
        else:
            # Kalman update
            y = z - self.H @ track.state
            S = self.H @ track.covariance @ self.H.T + self.R
            K = track.covariance @ self.H.T @ np.linalg.inv(S)
            
            track.state = track.state + K @ y
            track.covariance = (np.eye(4) - K @ self.H) @ track.covariance
        
        # Update track data
        track.positions.append(detection.center)
        track.timestamps.append(timestamp)
        track.confidences.append(detection.confidence)
        track.bboxes.append(detection.bbox)
        track.hits += 1
        track.misses = 0
        track.age += 1
    
    def _create_track(self, detection: Detection, timestamp: float):
        """Create new track from detection."""
        track = Track(track_id=self.next_id)
        self.next_id += 1
        
        track.positions.append(detection.center)
        track.timestamps.append(timestamp)
        track.confidences.append(detection.confidence)
        track.bboxes.append(detection.bbox)
        track.hits = 1
        track.age = 1
        
        # Initialize Kalman state
        track.state = np.array([
            detection.center[0], detection.center[1], 0, 0
        ])
        track.covariance = np.eye(4) * 100
        
        self.tracks.append(track)
    
    def _associate(
        self,
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using distance.
        
        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_dets: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Compute distance matrix
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        dist_matrix = np.zeros((n_tracks, n_dets))
        
        for i, track in enumerate(self.tracks):
            if track.state is None:
                dist_matrix[i, :] = self.max_distance + 1
                continue
                
            pred_pos = track.state[:2]
            
            for j, det in enumerate(detections):
                dist = np.sqrt(
                    (pred_pos[0] - det.center[0])**2 +
                    (pred_pos[1] - det.center[1])**2
                )
                dist_matrix[i, j] = dist
        
        # Greedy matching (simple but effective for single ball)
        matched = []
        unmatched_dets = list(range(n_dets))
        unmatched_tracks = list(range(n_tracks))
        
        while True:
            if len(unmatched_dets) == 0 or len(unmatched_tracks) == 0:
                break
            
            # Find minimum distance
            min_dist = float('inf')
            min_i, min_j = -1, -1
            
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        min_i, min_j = i, j
            
            if min_dist > self.max_distance:
                break
            
            matched.append((min_i, min_j))
            unmatched_tracks.remove(min_i)
            unmatched_dets.remove(min_j)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def get_best_track(self) -> Optional[Track]:
        """Get the most confident/longest track (likely the ball)."""
        confirmed = [t for t in self.tracks if t.is_confirmed]
        
        if not confirmed:
            return None
        
        # Score by length and average confidence
        def score(t):
            avg_conf = np.mean(t.confidences) if t.confidences else 0
            return len(t.positions) * avg_conf
        
        return max(confirmed, key=score)
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
