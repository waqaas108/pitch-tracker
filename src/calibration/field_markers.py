"""
Standard baseball field geometry for camera calibration.
All measurements in meters from home plate center.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


# Standard field dimensions (meters)
MOUND_DISTANCE = 18.44  # 60 ft 6 in
MOUND_HEIGHT = 0.254    # 10 inches
BASE_DISTANCE = 27.43   # 90 ft
PLATE_WIDTH = 0.432     # 17 inches
BATTER_BOX_DEPTH = 1.22 # 4 ft from plate center


@dataclass
class FieldMarkers:
    """
    Known 3D positions of field landmarks.
    Used for camera calibration via PnP (Perspective-n-Point).
    
    Coordinate system:
    - Origin: Home plate center, ground level
    - X: Toward first base (positive)
    - Y: Up (positive)
    - Z: Toward pitcher/center field (positive)
    """
    
    # Core calibration points (most useful)
    home_plate_center: np.ndarray = None
    home_plate_front: np.ndarray = None
    home_plate_back: np.ndarray = None
    
    pitching_rubber_center: np.ndarray = None
    pitching_rubber_left: np.ndarray = None
    pitching_rubber_right: np.ndarray = None
    
    first_base: np.ndarray = None
    second_base: np.ndarray = None
    third_base: np.ndarray = None
    
    # Batter's boxes
    left_box_front_inside: np.ndarray = None
    left_box_back_outside: np.ndarray = None
    right_box_front_inside: np.ndarray = None
    right_box_back_outside: np.ndarray = None
    
    def __post_init__(self):
        """Initialize with standard MLB dimensions."""
        # Home plate
        self.home_plate_center = np.array([0, 0, 0])
        self.home_plate_front = np.array([0, 0, -PLATE_WIDTH/2])
        self.home_plate_back = np.array([0, 0, PLATE_WIDTH/2])
        
        # Pitching rubber (0.61m wide = 24 inches)
        rubber_width = 0.61
        self.pitching_rubber_center = np.array([0, MOUND_HEIGHT, MOUND_DISTANCE])
        self.pitching_rubber_left = np.array([-rubber_width/2, MOUND_HEIGHT, MOUND_DISTANCE])
        self.pitching_rubber_right = np.array([rubber_width/2, MOUND_HEIGHT, MOUND_DISTANCE])
        
        # Bases
        self.first_base = np.array([BASE_DISTANCE * np.cos(np.pi/4), 0, BASE_DISTANCE * np.sin(np.pi/4)])
        self.second_base = np.array([0, 0, BASE_DISTANCE * np.sqrt(2)])
        self.third_base = np.array([-BASE_DISTANCE * np.cos(np.pi/4), 0, BASE_DISTANCE * np.sin(np.pi/4)])
        
        # Batter's boxes (approximate)
        box_width = 1.22  # 4 ft
        box_offset = 0.15  # offset from plate
        self.left_box_front_inside = np.array([-box_offset, 0, -BATTER_BOX_DEPTH/2])
        self.left_box_back_outside = np.array([-box_offset - box_width, 0, BATTER_BOX_DEPTH/2])
        self.right_box_front_inside = np.array([box_offset, 0, -BATTER_BOX_DEPTH/2])
        self.right_box_back_outside = np.array([box_offset + box_width, 0, BATTER_BOX_DEPTH/2])
    
    def get_point(self, name: str) -> np.ndarray:
        """Get a named point."""
        return getattr(self, name, None)
    
    def get_all_points(self) -> Dict[str, np.ndarray]:
        """Get all defined points as a dictionary."""
        points = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                val = getattr(self, attr)
                if isinstance(val, np.ndarray):
                    points[attr] = val
        return points


# Singleton instance with standard dimensions
STANDARD_FIELD_POINTS = FieldMarkers()


def get_calibration_point_options() -> Dict[str, str]:
    """
    Get human-readable names for calibration points.
    Returns dict of {internal_name: display_name}
    """
    return {
        'home_plate_center': 'Home Plate - Center',
        'home_plate_front': 'Home Plate - Front Point',
        'home_plate_back': 'Home Plate - Back Corner',
        'pitching_rubber_center': 'Pitching Rubber - Center',
        'pitching_rubber_left': 'Pitching Rubber - Left Edge',
        'pitching_rubber_right': 'Pitching Rubber - Right Edge',
        'first_base': 'First Base - Center',
        'second_base': 'Second Base - Center',
        'third_base': 'Third Base - Center',
        'left_box_front_inside': 'Left Batter Box - Front Inside Corner',
        'right_box_front_inside': 'Right Batter Box - Front Inside Corner',
    }
