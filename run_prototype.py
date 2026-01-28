"""
Main prototype script - generates data and trains models.
Run this to test the full pipeline on your 4060.
"""
import argparse
from pathlib import Path
import torch

def main():
    parser = argparse.ArgumentParser(description='Pitch Tracker ML Prototype')
    parser.add_argument('--step', type=str, choices=['all', 'data', 'regressor', 'classifier', 'demo'],
                        default='all', help='Which step to run')
    parser.add_argument('--samples', type=int, default=5000, help='Number of synthetic samples')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Paths
    data_dir = Path('./data/synthetic')
    regressor_dir = Path('./checkpoints/regressor')
    classifier_dir = Path('./checkpoints/classifier')
    
    # Check CUDA
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("CUDA not available, using CPU")
            args.device = 'cpu'
    
    if args.step in ['all', 'data']:
        print("\n" + "="*50)
        print("STEP 1: Generating Synthetic Data")
        print("="*50)
        
        from src.data.synthetic_generator import generate_dataset
        
        generate_dataset(
            n_samples=args.samples,
            output_dir=str(data_dir),
            seed=42
        )
    
    if args.step in ['all', 'regressor']:
        print("\n" + "="*50)
        print("STEP 2: Training Trajectory Regressor")
        print("="*50)
        
        from src.training.train_regressor import train_regressor
        
        train_regressor(
            dataset_dir=str(data_dir),
            output_dir=str(regressor_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    if args.step in ['all', 'classifier']:
        print("\n" + "="*50)
        print("STEP 3: Training Pitch Classifier")
        print("="*50)
        
        from src.training.train_classifier import train_classifier
        
        train_classifier(
            dataset_dir=str(data_dir),
            output_dir=str(classifier_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device
        )
    
    if args.step in ['all', 'demo']:
        print("\n" + "="*50)
        print("STEP 4: Running Demo Inference")
        print("="*50)
        
        run_demo(regressor_dir, classifier_dir, args.device)
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)


def run_demo(regressor_dir: Path, classifier_dir: Path, device: str):
    """Run inference on a few synthetic pitches."""
    import numpy as np
    
    from src.physics.trajectory import PitchParams, simulate_pitch
    from src.physics.camera import project_points, get_camera_presets, add_detection_noise
    from src.models.trajectory_regressor import TrajectoryRegressor
    from src.models.pitch_classifier import PitchClassifier
    from src.data.pitch_types import PITCH_PROFILES, sample_pitch_params, get_label_to_pitch_type
    
    # Load models
    regressor = TrajectoryRegressor()
    regressor.load_state_dict(torch.load(regressor_dir / 'best_model.pt')['model_state_dict'])
    regressor.to(device).eval()
    
    classifier = PitchClassifier(num_classes=len(PITCH_PROFILES))
    classifier.load_state_dict(torch.load(classifier_dir / 'best_model.pt')['model_state_dict'])
    classifier.to(device).eval()
    
    label_to_type = get_label_to_pitch_type()
    camera = get_camera_presets()['behind_mound_left']
    
    print("\nDemo: Simulating and predicting pitches\n")
    
    test_pitches = ['FF', 'CU', 'SL', 'CH']
    rng = np.random.default_rng(123)
    
    for pitch_type in test_pitches:
        # Generate a pitch
        params_dict = sample_pitch_params(pitch_type, rng)
        params = PitchParams(**params_dict)
        result = simulate_pitch(params)
        
        # Project to 2D
        positions = result.positions[::4]  # subsample
        traj_2d = project_points(positions, camera)
        noisy_2d, _ = add_detection_noise(traj_2d, noise_std=2.0, dropout_prob=0.05)
        
        # Pad to fixed length
        max_len = 50
        if len(noisy_2d) < max_len:
            pad = np.zeros((max_len - len(noisy_2d), 2))
            noisy_2d = np.vstack([noisy_2d, pad])
        else:
            noisy_2d = noisy_2d[:max_len]
        
        noisy_2d = np.nan_to_num(noisy_2d, 0.0)
        
        # Inference
        traj_tensor = torch.tensor(noisy_2d, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_metrics = regressor(traj_tensor)
            pred_logits = classifier(trajectory=traj_tensor, metrics=pred_metrics)
            pred_class = pred_logits.argmax(dim=1).item()
        
        pred_metrics = pred_metrics[0].cpu().numpy()
        pred_type = label_to_type[pred_class]
        
        print(f"Pitch: {PITCH_PROFILES[pitch_type].name}")
        print(f"  Ground Truth: {params.velocity:.1f} mph, H-Break: {result.horizontal_break:.1f}\", V-Break: {result.vertical_break:.1f}\"")
        print(f"  Predicted:    {pred_metrics[0]:.1f} mph, H-Break: {pred_metrics[1]:.1f}\", V-Break: {pred_metrics[2]:.1f}\"")
        print(f"  Predicted Type: {PITCH_PROFILES[pred_type].name} {'✓' if pred_type == pitch_type else '✗'}")
        print()


if __name__ == '__main__':
    main()
