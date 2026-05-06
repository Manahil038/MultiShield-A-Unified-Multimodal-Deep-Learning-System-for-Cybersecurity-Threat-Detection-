import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any

from src.utils.helpers import seed_everything
seed_everything(42)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str = 'binary',
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = 'cpu',
    save_dir: str = 'checkpoints',
    log_dir: str = 'runs/baseline_experiment'
) -> Dict[str, Any]:
    """
    End-to-end training loop with TensorBoard logging, early stopping, and testing.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model = model.to(device)
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Loss Function Choice
    if task_type == 'binary':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"Starting training on {device} for {epochs} epochs. Logging to {log_dir}")
    
    for epoch in range(epochs):
        # -----------------------------
        # 1. Training Phase
        # -----------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            
            if task_type == 'binary':
                loss = criterion(logits.squeeze(), batch_y.float())
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            else:
                loss = criterion(logits, batch_y.long())
                preds = torch.argmax(logits, dim=1)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)
            
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # -----------------------------
        # 2. Validation Phase
        # -----------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                
                if task_type == 'binary':
                    loss = criterion(logits.squeeze(), batch_y.float())
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
                else:
                    loss = criterion(logits, batch_y.long())
                    preds = torch.argmax(logits, dim=1)
                    
                val_loss += loss.item() * batch_X.size(0)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # -----------------------------
        # 3. TensorBoard Logging
        # -----------------------------
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Hyperparameters/LearningRate', current_lr, epoch)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # -----------------------------
        # 4. Early Stopping & Checkpointing
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\\n[!] Early stopping triggered at epoch {epoch+1}.")
                break
                
    # -----------------------------
    # 5. Final Evaluation (Test Set)
    # -----------------------------
    print("\\nLoading best model for Final Evaluation on held-out Test Set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            logits = model(batch_X)
            
            if task_type == 'binary':
                loss = criterion(logits.squeeze(), batch_y.float())
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            else:
                loss = criterion(logits, batch_y.long())
                preds = torch.argmax(logits, dim=1)
                
            test_loss += loss.item() * batch_X.size(0)
            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)
            
    test_loss /= len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    # Log Final Test Metrics to TensorBoard alongside Hyperparameters
    hparams = {
        'lr': lr, 
        'epochs': epochs, 
        'patience': patience, 
        'task_type': task_type
    }
    metrics = {
        'hparam/test_loss': test_loss,
        'hparam/test_acc': test_acc
    }
    writer.add_hparams(hparams, metrics)
    writer.close()
    
    print(f"=========================================")
    print(f"FINAL TEST SET RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"=========================================")
    
    return {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc
    }
