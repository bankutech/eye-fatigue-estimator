import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyre-ignore[21]
import config
from src.dataset import FatigueDataset, get_transforms
from src.model import FatigueModel

def train_model():
    # Set device to CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    if not os.path.exists(config.CSV_PATH):
        print(f"Dataset not found at {config.CSV_PATH}. Please run data_prep.py first.")
        return

    full_dataset = FatigueDataset(
        csv_file=config.CSV_PATH,
        img_dir=config.PROCESSED_DATA_DIR,
        transform=None # Transform applied per subset
    )

    # Split into train and validation (80/20)
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        print("Dataset is empty.")
        return
        
    train_size = int(0.8 * dataset_size)
    
    # Use sequential split to prevent data leakage from consecutive frames
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # We need to apply different transforms to train/val datasets
    # A simple hack is to wrap them
    class TransformWrap(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            img, score = self.subset[index]
            # In random_split, subset[idx] goes to full_dataset[idx] which returns raw image (if full_dataset.transform=None)
            # Wait, full_dataset.__getitem__ returns (image, score). 
            # We should pass transform to full_dataset but it applies to all.
            # To fix this, we apply transforms here.
            # Let's override full_dataset to return PIL image, and apply transform here.
            # However, subset[idx] returns already unpacked tuple.
            return self.transform(img), score
            
        def __len__(self):
            return len(self.subset)
            
    # Apply transforms
    train_dataset = TransformWrap(train_dataset, get_transforms(is_train=True))
    val_dataset = TransformWrap(val_dataset, get_transforms(is_train=False))

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, Loss, Optimizer
    model = FatigueModel(pretrained=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    best_model_path = os.path.join(config.MODELS_DIR, "best_model.pth")

    # Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        progress_bar = tqdm(train_loader, desc="Training")
        
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        train_loss /= len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss (MSE): {val_loss:.4f}")
        
        # Early stopping / Model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model to {best_model_path}")

    print("Training complete!")

if __name__ == "__main__":
    train_model()
