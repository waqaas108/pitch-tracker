"""
Training script for the trajectory initializer network.
This network learns to predict good starting points for the physics optimizer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from ..data.stereo_generator import load_stereo_dataset, StereoSample
from ..physics.trajectory_fitter import TrajectoryInitializer


class StereoDataset(Dataset):
    """PyTorch dataset for stereo samples."""
    
    def __init__(self, samples: list, max_seq_len: int = 50):
        self.samples = samples
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Stack observations: (wide_x, wide_y, ultra_x, ultra_y, depth)
        n = len(sample.wide_2d)
        obs = np.zeros((self.max_seq_len, 5), dtype=np.float32)
        
        n_use = min(n, self.max_seq_len)
        obs[:n_use, 0] = sample.wide_2d[:n_use, 0]
        obs[:n_use, 1] = sample.wide_2d[:n_use, 1]
        obs[:n_use, 2] = sample.ultrawide_2d[:n_use, 0]
        obs[:n_use, 3] = sample.ultrawide_2d[:n_use, 1]
        obs[:n_use, 4] = sample.depths[:n_use]
        
        # Replace NaN with 0 (will be handled by model)
        obs = np.nan_to_num(obs, 0.0)
        
        # Target: physics parameters
        # Convert spin axis to spherical coordinates
        sx, sy, sz = sample.spin_axis
        spin_theta = np.arctan2(sz, sx)
        spin_phi = np.arccos(np.clip(sy, -1, 1))
        
        target = np.array([
            sample.velocity,
            sample.spin_rate,
            spin_theta,
            spin_phi,
            sample.release_angle_h,
            sample.release_angle_v,
        ], dtype=np.float32)
        
        return torch.from_numpy(obs), torch.from_numpy(target)


def train_initializer(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cuda'
):
    """
    Train the trajectory initializer network.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    samples, metadata = load_stereo_dataset(dataset_dir)
    
    # Split train/val
    n_val = int(len(samples) * 0.1)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    train_dataset = StereoDataset(train_samples)
    val_dataset = StereoDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = TrajectoryInitializer().to(device)
    
    # Loss: weighted MSE for different parameter scales
    param_weights = torch.tensor([
        1.0,    # velocity
        0.001,  # spin_rate (larger values)
        1.0,    # spin_theta
        1.0,    # spin_phi
        1.0,    # angle_h
        1.0,    # angle_v
    ]).to(device)
    
    def weighted_mse(pred, target):
        diff = (pred - target) ** 2
        return (diff * param_weights).mean()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"Training initializer on {len(train_samples)} samples...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        
        for obs, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            obs = obs.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            pred = model(obs)
            loss = weighted_mse(pred, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for obs, target in val_loader:
                obs = obs.to(device)
                target = target.to(device)
                
                pred = model(obs)
                loss = weighted_mse(pred, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / 'best_initializer.pt')
    
    # Save final model and history
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, output_path / 'final_initializer.pt')
    
    with open(output_path / 'initializer_history.json', 'w') as f:
        json.dump(history, f)
    
    print(f"Training complete. Best val_loss: {best_val_loss:.4f}")
    
    return model
