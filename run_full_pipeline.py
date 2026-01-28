"""
Full pitch analysis pipeline demo.
Shows the complete flow: Video → Detection → Tracking → Physics → Metrics

This script demonstrates:
1. Processing real video (if available)
2. Processing synthetic video (for testing)
3. End-to-end analysis with all components
"""
import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description='Full Pitch Analysis Pipeline')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--wide-video', type=str, help='Path to wide camera video (stereo)')
    parser.add_argument('--ultra-video', type=str, help='Path to ultrawide camera video (stereo)')
    parser.add_argument('--synthetic', action='store_true', help='Run on synthetic data')
    parser.add_argument('--model-dir', type=str, default='./checkpoints/stereo',
                        help='Path to trained models')
    parser.add_argument('--calibration', type=str, help='Path to camera calibration JSON')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load calibration if provided
    calibration = None
    if args.calibration:
        from src.calibration.camera_calibrator import CalibrationResult
        calibration = CalibrationResult.load(args.calibration)
        print(f"Loaded calibration from {args.calibration}")
    
    if args.synthetic:
        run_synthetic_demo(args.model_dir, args.device)
    elif args.video:
        run_single_video(args.video, args.model_dir, args.device, calibration)
    elif args.wide_video and args.ultra_video:
        run_stereo_video(args.wide_video, args.ultra_video, args.model_dir, args.device, calibration)
    else:
        print("Running synthetic demo (no video provided)")
        print("Use --video <path> for single camera or --wide-video/--ultra-video for stereo")
        print("Use --calibration <path> to load camera calibration")
        run_synthetic_demo(args.model_dir, args.device)


def run_synthetic_demo(model_dir: str, device: str):
    """
    Demo using synthetic data to test the full pipeline.
    Simulates what would happen with real video input.
    """
    print("\n" + "="*60)
    print("SYNTHETIC PIPELINE DEMO")
    print("="*60)
    
    from src.pipeline.pitch_analyzer import PitchAnalyzer
    from src.physics.trajectory import PitchParams, simulate_pitch
    from src.physics.stereo_camera import (
        get_stereo_camera_presets, project_to_stereo, add_stereo_noise
    )
    from src.data.pitch_types import PITCH_PROFILES, sample_pitch_params
    
    # Initialize analyzer
    analyzer = PitchAnalyzer(device=device)
    
    # Load models
    model_path = Path(model_dir)
    if (model_path / 'best_initializer.pt').exists():
        analyzer.load_models(model_dir)
    else:
        print(f"Warning: No models found in {model_dir}")
        print("Run 'python run_stereo_prototype.py --step all' first")
    
    # Get camera rig
    rig = get_stereo_camera_presets()['behind_high']
    analyzer.camera_rig = rig
    
    rng = np.random.default_rng(789)
    
    print("\nSimulating pitches and running full analysis...\n")
    
    test_pitches = ['FF', 'CU', 'SL', 'CH', 'SI']
    
    for pitch_type in test_pitches:
        # Generate ground truth pitch
        params_dict = sample_pitch_params(pitch_type, rng)
        params = PitchParams(**params_dict)
        result = simulate_pitch(params)
        
        # Subsample to 60fps
        indices = np.arange(0, len(result.positions), 17)
        positions = result.positions[indices]
        timestamps = result.times[indices]
        
        # Project to stereo cameras (simulating detection output)
        wide_2d, ultra_2d, depths = project_to_stereo(positions, rig)
        
        # Add realistic noise (simulating detection errors)
        noisy_wide, noisy_ultra, noisy_depths, _ = add_stereo_noise(
            wide_2d, ultra_2d, depths,
            pixel_noise_std=3.0,  # Slightly more noise than training
            depth_noise_std=0.2,
            dropout_prob=0.08,
            rng=rng
        )
        
        # Run analysis
        analysis = analyzer.analyze_trajectory(
            noisy_wide, noisy_ultra, noisy_depths, timestamps
        )
        
        if analysis:
            # Compare to ground truth
            vel_err = abs(analysis.velocity - params.velocity)
            h_err = abs(analysis.horizontal_break - result.horizontal_break)
            v_err = abs(analysis.vertical_break - result.vertical_break)
            
            type_match = "✓" if analysis.pitch_type == pitch_type else "✗"
            
            print(f"{PITCH_PROFILES[pitch_type].name}")
            print(f"  Ground Truth: {params.velocity:.1f} mph, "
                  f"H: {result.horizontal_break:.1f}\", V: {result.vertical_break:.1f}\"")
            print(f"  Predicted:    {analysis.velocity:.1f} mph, "
                  f"H: {analysis.horizontal_break:.1f}\", V: {analysis.vertical_break:.1f}\"")
            print(f"  Errors:       vel={vel_err:.1f} mph, H={h_err:.1f}\", V={v_err:.1f}\"")
            print(f"  Classified:   {analysis.pitch_type_name} {type_match} "
                  f"(conf: {analysis.pitch_type_confidence:.0%})")
            print()
        else:
            print(f"{PITCH_PROFILES[pitch_type].name}: Analysis failed\n")


def run_single_video(video_path: str, model_dir: str, device: str, calibration=None):
    """
    Analyze a single-camera video.
    """
    print("\n" + "="*60)
    print("SINGLE CAMERA VIDEO ANALYSIS")
    print("="*60)
    
    from src.pipeline.pitch_analyzer import PitchAnalyzer
    
    analyzer = PitchAnalyzer(device=device)
    analyzer.load_models(model_dir)
    
    # Apply calibration if provided
    if calibration:
        analyzer.camera_rig = calibration.to_stereo_rig()
        print("Using provided camera calibration")
    else:
        print("Warning: No calibration provided, using default camera position")
        print("Run 'python calibrate_camera.py' for better accuracy")
    
    print(f"\nAnalyzing: {video_path}")
    
    result = analyzer.analyze_video(video_path)
    
    if result:
        print("\n" + "-"*40)
        print("ANALYSIS RESULT")
        print("-"*40)
        print(result)
        print(f"\nFit quality: {result.fit_residual:.2f}")
        print(f"Trajectory points: {result.trajectory_length}")
    else:
        print("\nAnalysis failed - could not detect ball trajectory")
        print("Tips:")
        print("  - Ensure ball is visible in video")
        print("  - Try lowering detection confidence threshold")
        print("  - Consider fine-tuning YOLO on baseball data")


def run_stereo_video(wide_path: str, ultra_path: str, model_dir: str, device: str, calibration=None):
    """
    Analyze synchronized stereo video pair.
    """
    print("\n" + "="*60)
    print("STEREO VIDEO ANALYSIS")
    print("="*60)
    
    from src.pipeline.pitch_analyzer import PitchAnalyzer
    
    analyzer = PitchAnalyzer(device=device)
    analyzer.load_models(model_dir)
    
    # Apply calibration if provided
    if calibration:
        analyzer.camera_rig = calibration.to_stereo_rig()
        print("Using provided camera calibration")
    
    print(f"\nWide camera: {wide_path}")
    print(f"Ultrawide camera: {ultra_path}")
    
    result = analyzer.analyze_stereo_video(wide_path, ultra_path)
    
    if result:
        print("\n" + "-"*40)
        print("ANALYSIS RESULT")
        print("-"*40)
        print(result)
        print(f"\nFit quality: {result.fit_residual:.2f}")
        print(f"Trajectory points: {result.trajectory_length}")
    else:
        print("\nAnalysis failed - could not detect ball trajectory in one or both videos")
        print("Ensure both videos are synchronized and ball is visible")


if __name__ == '__main__':
    main()
