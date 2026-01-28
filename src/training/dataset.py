"""
PyTorch Dataset for trajectory data.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import json

from ..data.synthetic_generator import SyntheticSample, load_dataset


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory regression and classification.
    """
    
    def __init__(
        self,
        samples: List[SyntheticSample],
        max_seq_len: int = 50,
        return_metrics: bool = True,
        return_class: bool = True
    ):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.return_metrics = return_metrics
        self.return_class = return_class
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        
        # Get trajectory and pad/truncate
        traj = sample.trajectory_2d.copy()
        seq_len = len(traj)
        
        if seq_len > self.max_seq_len:
            # Truncate
            traj = traj[:self.max_seq_len]
            seq_len = self.max_seq_len
        elif seq_len < self.max_seq_len:
            # Pad with zeros
            pad = np.zeros((self.max_seq_len - seq_len, 2))
            traj = np.vstack([traj, pad])
        
        # Replace NaN with 0 for now (model handles interpolation)
        traj = np.nan_to_num(traj, 0.0)
        
        result = {
            'trajectory': torch.tensor(traj, dtype=torch.float32),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
        }
        
        if self.return_metrics:
            metrics = torch.tensor([
                sample.velocity,
                sample.horizontal_break,
                sample.vertical_break,
                sample.plate_location[0],
                sample.plate_location[1],
            ], dtype=torch.float32)
            result['metrics'] = metrics
        
        if self.return_class:
            result['pitch_type'] = torch.tensor(sample.pitch_type_label, dtype=torch.long)
        
        return result


def create_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    max_seq_len: int = 50,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders from a dataset directory.
    """
    samples, metadata = load_dataset(dataset_dir)
    
    # Shuffle and split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    
    n_train = int(len(samples) * train_split)
    n_val = int(len(samples) * val_split)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    
    train_ds = TrajectoryDataset(train_samples, max_seq_len)
    val_ds = TrajectoryDataset(val_samples, max_seq_len)
    test_ds = TrajectoryDataset(test_samples, max_seq_len)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, metadata
