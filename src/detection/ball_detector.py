"""
Ball detection using YOLOv8.
Detects baseball in video frames and returns bounding boxes.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch


@dataclass
class Detection:
    """Single ball detection."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[float, float]  # center x, y
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class BallDetector:
    """
    YOLO-based baseball detector.
    
    Uses YOLOv8 with optional fine-tuning for baseball detection.
    Falls back to sports ball class from COCO if no custom model.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize detector.
        
        Args:
            model_path: Path to custom YOLO model weights (optional)
            confidence_threshold: Minimum confidence for detections
            device: 'cuda' or 'cpu'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.model_path = model_path
        
        # COCO class ID for sports ball
        self.sports_ball_class = 32
        
    def load_model(self):
        """Load YOLO model (lazy loading)."""
        if self.model is not None:
            return
            
        try:
            from ultralytics import YOLO
            
            if self.model_path:
                # Custom fine-tuned model
                self.model = YOLO(self.model_path)
                self.custom_model = True
            else:
                # Default YOLOv8 with COCO weights
                self.model = YOLO('yolov8n.pt')  # nano for speed
                self.custom_model = False
                
            print(f"Loaded YOLO model: {'custom' if self.custom_model else 'yolov8n (COCO)'}")
            
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect balls in a single frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            List of Detection objects
        """
        self.load_model()
        
        # Run inference
        results = self.model(frame, verbose=False, device=self.device)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                # Filter by class and confidence
                if self.custom_model:
                    # Custom model: class 0 is baseball
                    if cls != 0 or conf < self.confidence_threshold:
                        continue
                else:
                    # COCO model: class 32 is sports ball
                    if cls != self.sports_ball_class or conf < self.confidence_threshold:
                        continue
                
                # Get bbox
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    center=center
                ))
        
        # Sort by confidence, return best detection
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect balls in multiple frames (batched for efficiency).
        
        Args:
            frames: List of BGR images
            
        Returns:
            List of detection lists, one per frame
        """
        self.load_model()
        
        # Run batched inference
        results = self.model(frames, verbose=False, device=self.device)
        
        all_detections = []
        
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                if self.custom_model:
                    if cls != 0 or conf < self.confidence_threshold:
                        continue
                else:
                    if cls != self.sports_ball_class or conf < self.confidence_threshold:
                        continue
                
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                frame_detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    center=center
                ))
            
            frame_detections.sort(key=lambda d: d.confidence, reverse=True)
            all_detections.append(frame_detections)
        
        return all_detections


class ColorBasedDetector:
    """
    Simple color-based ball detector as fallback.
    Useful when YOLO fails or for quick prototyping.
    """
    
    def __init__(
        self,
        min_radius: int = 5,
        max_radius: int = 50,
        white_threshold: int = 200
    ):
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.white_threshold = white_threshold
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect white circular objects (baseballs)."""
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold for white objects
        _, thresh = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
        
        # Find circles using Hough transform
        circles = cv2.HoughCircles(
            thresh,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=20,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )
        
        detections = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for circle in circles[0]:
                cx, cy, r = circle
                
                # Create bbox from circle
                x1 = cx - r
                y1 = cy - r
                x2 = cx + r
                y2 = cy + r
                
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=0.5,  # Fixed confidence for color-based
                    center=(float(cx), float(cy))
                ))
        
        return detections
