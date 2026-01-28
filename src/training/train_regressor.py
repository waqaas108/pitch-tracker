"""
Training script for trajectory regressor.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from typing import Optional

from ..models.trajectory_regressor import TrajectoryRegressor
from .dataset import create_dataloaders


def train_regressor(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    device: str = 'cuda'
) -> dict:
    """
    Train the trajectory regressor model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Training on {device}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        dataset_dir, batch_size=batch_size
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = TrajectoryRegressor(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=5
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            traj = batch['trajectory'].to(device)
            metrics = batch['metrics'].to(device)
            seq_lens = batch['seq_len']
            
            optimizer.zero_grad()
            pred = model(traj, seq_lens)
            loss = criterion(pred, metrics)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                traj = batch['trajectory'].to(device)
                metrics = batch['metrics'].to(device)
                seq_lens = batch['seq_len']
                
                pred = model(traj, seq_lens)
                loss = criterion(pred, metrics)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_path / 'best_model.pt')
    
    # Final evaluation
    model.load_state_dict(torch.load(output_path / 'best_model.pt')['model_state_dict'])
    model.eval()
    
    test_metrics = evaluate_regressor(model, test_loader, device)
    
    # Save results
    results = {
        'history': history,
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return results


def evaluate_regressor(model: nn.Module, loader: DataLoader, device: str) -> dict:
    """Evaluate regressor on a dataset."""
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            traj = batch['trajectory'].to(device)
            metrics = batch['metrics'].to(device)
            seq_lens = batch['seq_len']
            
            pred = model(traj, seq_lens)
            all_preds.append(pred.cpu())
            all_targets.append(metrics.cpu())
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = ((preds - targets) ** 2).mean(dim=0)
    mae = (preds - targets).abs().mean(dim=0)
    
    metric_names = ['velocity', 'h_break', 'v_break', 'plate_x', 'plate_y']
    
    results = {}
    for i, name in enumerate(metric_names):
        results[f'{name}_mse'] = mse[i].item()
        results[f'{name}_mae'] = mae[i].item()
    
    results['total_mse'] = mse.mean().item()
    results['total_mae'] = mae.mean().item()
    
    # Velocity-specific (most important)
    results['velocity_rmse'] = (mse[0] ** 0.5).item()
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='./checkpoints/regressor')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    train_regressor(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
