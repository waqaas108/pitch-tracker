"""
Training script for pitch type classifier.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from ..models.pitch_classifier import PitchClassifier
from .dataset import create_dataloaders


def train_classifier(
    dataset_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    use_trajectory: bool = True,
    use_metrics: bool = True,
    device: str = 'cuda'
) -> dict:
    """
    Train the pitch type classifier.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Training on {device}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        dataset_dir, batch_size=batch_size
    )
    
    num_classes = len(metadata['label_mapping'])
    print(f"Number of classes: {num_classes}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create model
    model = PitchClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        use_trajectory=use_trajectory,
        use_metrics=use_metrics
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            traj = batch['trajectory'].to(device) if use_trajectory else None
            metrics = batch['metrics'].to(device) if use_metrics else None
            labels = batch['pitch_type'].to(device)
            
            optimizer.zero_grad()
            logits = model(trajectory=traj, metrics=metrics)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                traj = batch['trajectory'].to(device) if use_trajectory else None
                metrics = batch['metrics'].to(device) if use_metrics else None
                labels = batch['pitch_type'].to(device)
                
                logits = model(trajectory=traj, metrics=metrics)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.3f} | Val Loss={val_loss:.4f}, Acc={val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, output_path / 'best_model.pt')
    
    # Final evaluation
    model.load_state_dict(torch.load(output_path / 'best_model.pt')['model_state_dict'])
    
    test_metrics = evaluate_classifier(
        model, test_loader, device, 
        use_trajectory=use_trajectory, 
        use_metrics=use_metrics,
        label_mapping=metadata['label_mapping']
    )
    
    # Save results
    results = {
        'history': history,
        'test_metrics': test_metrics,
        'best_val_acc': best_val_acc,
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  Per-class accuracy:")
    for cls, acc in test_metrics['per_class_accuracy'].items():
        print(f"    {cls}: {acc:.3f}")
    
    return results


def evaluate_classifier(
    model: nn.Module, 
    loader: DataLoader, 
    device: str,
    use_trajectory: bool = True,
    use_metrics: bool = True,
    label_mapping: dict = None
) -> dict:
    """Evaluate classifier on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            traj = batch['trajectory'].to(device) if use_trajectory else None
            metrics = batch['metrics'].to(device) if use_metrics else None
            labels = batch['pitch_type']
            
            logits = model(trajectory=traj, metrics=metrics)
            preds = logits.argmax(dim=1).cpu()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    preds = torch.cat(all_preds, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    # Overall accuracy
    accuracy = (preds == labels).mean()
    
    # Per-class accuracy
    per_class_acc = {}
    if label_mapping:
        inv_mapping = {v: k for k, v in label_mapping.items()}
        for label_idx in np.unique(labels):
            mask = labels == label_idx
            class_acc = (preds[mask] == labels[mask]).mean()
            class_name = inv_mapping.get(label_idx, str(label_idx))
            per_class_acc[class_name] = float(class_acc)
    
    # Confusion matrix (as nested dict)
    confusion = {}
    if label_mapping:
        inv_mapping = {v: k for k, v in label_mapping.items()}
        for true_idx in np.unique(labels):
            true_name = inv_mapping.get(true_idx, str(true_idx))
            confusion[true_name] = {}
            for pred_idx in np.unique(preds):
                pred_name = inv_mapping.get(pred_idx, str(pred_idx))
                count = ((labels == true_idx) & (preds == pred_idx)).sum()
                confusion[true_name][pred_name] = int(count)
    
    return {
        'accuracy': float(accuracy),
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion,
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='./checkpoints/classifier')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-trajectory', action='store_true')
    parser.add_argument('--no-metrics', action='store_true')
    
    args = parser.parse_args()
    
    train_classifier(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        use_trajectory=not args.no_trajectory,
        use_metrics=not args.no_metrics
    )
