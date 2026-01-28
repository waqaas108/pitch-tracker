"""
Stereo pitch tracking prototype.
Uses physics-constrained optimization with neural network initialization.

This is the improved approach based on research findings:
1. Stereo cameras provide depth information (like iPhone wide + ultrawide)
2. Physics optimizer fits trajectory parameters to observations
3. Neural net provides fast initialization for the optimizer
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import time

def main():
    parser = argparse.ArgumentParser(description='Stereo Pitch Tracker Prototype')
    parser.add_argument('--step', type=str, 
                        choices=['all', 'data', 'train', 'evaluate', 'demo'],
                        default='all', help='Which step to run')
    parser.add_argument('--samples', type=int, default=10000, 
                        help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Training epochs for initializer')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Paths
    data_dir = Path('./data/stereo')
    model_dir = Path('./checkpoints/stereo')
    
    # Check CUDA
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")
            args.device = 'cpu'
    
    if args.step in ['all', 'data']:
        print("\n" + "="*60)
        print("STEP 1: Generating Stereo Synthetic Data")
        print("="*60)
        
        from src.data.stereo_generator import generate_stereo_dataset
        
        generate_stereo_dataset(
            n_samples=args.samples,
            output_dir=str(data_dir),
            seed=42
        )
    
    if args.step in ['all', 'train']:
        print("\n" + "="*60)
        print("STEP 2: Training Trajectory Initializer")
        print("="*60)
        
        from src.training.train_initializer import train_initializer
        
        train_initializer(
            dataset_dir=str(data_dir),
            output_dir=str(model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    if args.step in ['all', 'evaluate']:
        print("\n" + "="*60)
        print("STEP 3: Evaluating Physics Fitter")
        print("="*60)
        
        evaluate_fitter(data_dir, model_dir, args.device)
    
    if args.step in ['all', 'demo']:
        print("\n" + "="*60)
        print("STEP 4: Running Demo")
        print("="*60)
        
        run_demo(model_dir, args.device)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


def evaluate_fitter(data_dir: Path, model_dir: Path, device: str):
    """Evaluate the physics fitter on held-out data."""
    from src.data.stereo_generator import load_stereo_dataset
    from src.physics.trajectory_fitter import (
        fit_trajectory_to_observations, fit_with_neural_init,
        TrajectoryInitializer
    )
    from src.physics.stereo_camera import get_stereo_camera_presets
    
    # Load data
    samples, metadata = load_stereo_dataset(str(data_dir))
    test_samples = samples[:100]  # Use first 100 for evaluation
    
    # Load initializer
    initializer = TrajectoryInitializer().to(device)
    ckpt = torch.load(model_dir / 'best_initializer.pt', weights_only=True)
    initializer.load_state_dict(ckpt['model_state_dict'])
    initializer.eval()
    
    camera_presets = get_stereo_camera_presets()
    
    # Metrics
    velocity_errors = []
    break_h_errors = []
    break_v_errors = []
    fit_times_with_init = []
    fit_times_without_init = []
    
    print(f"\nEvaluating on {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        rig = camera_presets[sample.camera_preset]
        
        # Fit with neural initialization
        t0 = time.time()
        result_with_init = fit_with_neural_init(
            sample.wide_2d, sample.ultrawide_2d, sample.depths,
            sample.timestamps, rig, initializer, device
        )
        fit_times_with_init.append(time.time() - t0)
        
        # Fit without initialization (for comparison)
        t0 = time.time()
        result_no_init = fit_trajectory_to_observations(
            sample.wide_2d, sample.ultrawide_2d, sample.depths,
            sample.timestamps, rig
        )
        fit_times_without_init.append(time.time() - t0)
        
        if result_with_init.success and result_with_init.params is not None:
            velocity_errors.append(abs(result_with_init.params.velocity - sample.velocity))
            break_h_errors.append(abs(result_with_init.result.horizontal_break - sample.horizontal_break))
            break_v_errors.append(abs(result_with_init.result.vertical_break - sample.vertical_break))
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(test_samples)}")
    
    # Report results
    print("\n" + "-"*40)
    print("EVALUATION RESULTS")
    print("-"*40)
    print(f"Samples evaluated: {len(velocity_errors)}/{len(test_samples)}")
    print(f"\nVelocity Error:")
    print(f"  Mean: {np.mean(velocity_errors):.2f} mph")
    print(f"  Std:  {np.std(velocity_errors):.2f} mph")
    print(f"  Max:  {np.max(velocity_errors):.2f} mph")
    print(f"\nHorizontal Break Error:")
    print(f"  Mean: {np.mean(break_h_errors):.2f} inches")
    print(f"  Std:  {np.std(break_h_errors):.2f} inches")
    print(f"\nVertical Break Error:")
    print(f"  Mean: {np.mean(break_v_errors):.2f} inches")
    print(f"  Std:  {np.std(break_v_errors):.2f} inches")
    print(f"\nFit Time (with neural init): {np.mean(fit_times_with_init)*1000:.1f} ms")
    print(f"Fit Time (without init):     {np.mean(fit_times_without_init)*1000:.1f} ms")
    print(f"Speedup: {np.mean(fit_times_without_init)/np.mean(fit_times_with_init):.1f}x")


def run_demo(model_dir: Path, device: str):
    """Run interactive demo on sample pitches."""
    from src.physics.trajectory import PitchParams, simulate_pitch
    from src.physics.stereo_camera import (
        get_stereo_camera_presets, project_to_stereo, add_stereo_noise
    )
    from src.physics.trajectory_fitter import (
        fit_with_neural_init, TrajectoryInitializer
    )
    from src.data.pitch_types import PITCH_PROFILES, sample_pitch_params
    
    # Load initializer
    initializer = TrajectoryInitializer().to(device)
    ckpt = torch.load(model_dir / 'best_initializer.pt', weights_only=True)
    initializer.load_state_dict(ckpt['model_state_dict'])
    initializer.eval()
    
    rig = get_stereo_camera_presets()['behind_high']
    rng = np.random.default_rng(456)
    
    print("\nDemo: Physics-Constrained Pitch Fitting")
    print("="*50)
    
    test_pitches = ['FF', 'CU', 'SL', 'CH', 'SI', 'FC']
    
    for pitch_type in test_pitches:
        # Generate ground truth pitch
        params_dict = sample_pitch_params(pitch_type, rng)
        params = PitchParams(**params_dict)
        result = simulate_pitch(params)
        
        # Subsample to 60fps
        indices = np.arange(0, len(result.positions), 17)  # ~60fps from 1000Hz sim
        positions = result.positions[indices]
        timestamps = result.times[indices]
        
        # Project to stereo cameras
        wide_2d, ultra_2d, depths = project_to_stereo(positions, rig)
        
        # Add noise
        noisy_wide, noisy_ultra, noisy_depths, _ = add_stereo_noise(
            wide_2d, ultra_2d, depths,
            pixel_noise_std=2.0, depth_noise_std=0.15, dropout_prob=0.05, rng=rng
        )
        
        # Fit trajectory
        fit_result = fit_with_neural_init(
            noisy_wide, noisy_ultra, noisy_depths,
            timestamps, rig, initializer, device
        )
        
        if fit_result.success and fit_result.params is not None:
            pred_params = fit_result.params
            pred_result = fit_result.result
            
            vel_err = abs(pred_params.velocity - params.velocity)
            h_err = abs(pred_result.horizontal_break - result.horizontal_break)
            v_err = abs(pred_result.vertical_break - result.vertical_break)
            
            status = "✓" if vel_err < 2 and h_err < 1 and v_err < 1 else "~"
            
            print(f"\n{PITCH_PROFILES[pitch_type].name} {status}")
            print(f"  Velocity:  {params.velocity:.1f} → {pred_params.velocity:.1f} mph (err: {vel_err:.1f})")
            print(f"  H-Break:   {result.horizontal_break:.1f} → {pred_result.horizontal_break:.1f}\" (err: {h_err:.1f})")
            print(f"  V-Break:   {result.vertical_break:.1f} → {pred_result.vertical_break:.1f}\" (err: {v_err:.1f})")
            print(f"  Spin Rate: {params.spin_rate:.0f} → {pred_params.spin_rate:.0f} rpm")
            print(f"  Residual:  {fit_result.residual:.2f}")
        else:
            print(f"\n{PITCH_PROFILES[pitch_type].name} ✗ (fit failed)")


if __name__ == '__main__':
    main()
