#!/usr/bin/env python3
"""
Camera calibration tool.
Run this to calibrate your camera setup before analyzing pitches.

Usage:
    # Interactive calibration from video frame
    python calibrate_camera.py --video pitch.mp4 --output calibration.json
    
    # From image file
    python calibrate_camera.py --image frame.jpg --output calibration.json
    
    # Quick calibration with known camera position
    python calibrate_camera.py --position 0,2.5,-2 --look-at 0,1,9 --output calibration.json
"""
import argparse
import numpy as np
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    
    # Input options
    parser.add_argument('--video', type=str, help='Video file to extract frame from')
    parser.add_argument('--image', type=str, help='Image file for calibration')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to use from video')
    
    # Quick calibration (skip interactive)
    parser.add_argument('--position', type=str, 
                        help='Camera position as x,y,z (meters). E.g., "0,2.5,-2"')
    parser.add_argument('--look-at', type=str,
                        help='Point camera looks at as x,y,z. E.g., "0,1,9"')
    
    # Output
    parser.add_argument('--output', type=str, default='calibration.json',
                        help='Output calibration file')
    
    args = parser.parse_args()
    
    if args.position and args.look_at:
        # Quick calibration from known position
        quick_calibrate(args.position, args.look_at, args.output)
    elif args.video or args.image:
        # Interactive calibration
        interactive_calibrate(args.video, args.image, args.frame, args.output)
    else:
        print("Provide --video/--image for interactive calibration")
        print("Or --position and --look-at for quick calibration")
        print("\nExample positions:")
        print("  Behind mound, high:  --position 0,3,-2 --look-at 0,1,9")
        print("  Behind mound, low:   --position 0,1.8,-1.5 --look-at 0,0.8,9")
        print("  Side view (1B):      --position 12,2,9 --look-at 0,1.2,9")
        print("  Behind catcher:      --position 0,1.5,20 --look-at 0,1.5,0")


def quick_calibrate(position_str: str, look_at_str: str, output_path: str):
    """Create calibration from known camera position."""
    from src.calibration.camera_calibrator import CalibrationResult
    
    # Parse position
    position = np.array([float(x) for x in position_str.split(',')])
    look_at = np.array([float(x) for x in look_at_str.split(',')])
    
    # Compute rotation matrix (look-at)
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)
    
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    true_up = np.cross(right, forward)
    
    rotation = np.column_stack([right, true_up, -forward])
    
    result = CalibrationResult(
        position=position,
        rotation=rotation,
        focal_length=1400.0,  # Default iPhone focal length
        principal_point=(960, 540),
        reprojection_error=0.0,
        num_points_used=0
    )
    
    result.save(output_path)
    
    print(f"Created calibration:")
    print(f"  Position: {position}")
    print(f"  Look at: {look_at}")
    print(f"  Saved to: {output_path}")


def interactive_calibrate(video_path: str, image_path: str, frame_num: int, output_path: str):
    """Interactive calibration from video/image."""
    import cv2
    from src.calibration.camera_calibrator import CameraCalibrator
    
    # Load frame
    if video_path:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Could not read frame {frame_num} from {video_path}")
            return
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load image: {image_path}")
            return
    
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run interactive calibration
    calibrator = CameraCalibrator(image_size=(frame.shape[1], frame.shape[0]))
    result = calibrator.calibrate_interactive(frame)
    
    if result:
        result.save(output_path)
        print(f"\nCalibration saved to: {output_path}")
        print(f"Camera position: {result.position}")
        print(f"Reprojection error: {result.reprojection_error:.2f} pixels")
    else:
        print("Calibration cancelled or failed")


if __name__ == '__main__':
    main()
