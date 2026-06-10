import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, random_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyre-ignore[21]
import config
from src.dataset import FatigueDataset, get_transforms
from src.model import FatigueModel

def evaluate_model():
    device = torch.device("cpu")
    print(f"Evaluating on device: {device}")
    
    best_model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Trained model not found. Please run train.py first.")
        return

    # Initialize model and load weights
    model = FatigueModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load dataset (we evaluate on the same validation split by fixing the random seed, or just evaluate on everything for demo purposes)
    # For a real project, we'd have a separate test set. Here we evaluate on the whole dataset to see fit, 
    # or you can use train_test_split with a fixed random_state.
    full_dataset = FatigueDataset(
        csv_file=config.CSV_PATH,
        img_dir=config.PROCESSED_DATA_DIR,
        transform=get_transforms(is_train=False)
    )
    
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    indices = list(range(dataset_size))
    val_indices = indices[train_size:]
    
    from torch.utils.data import Subset
    val_dataset = Subset(full_dataset, val_indices)
    
    loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    actuals = []
    predictions = []
    
    print("Running inference on the dataset...")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            
            actuals.extend(targets.numpy().flatten())
            predictions.extend(outputs.cpu().numpy().flatten())
            
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    
    # Calculate Metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.5, color='blue')
    plt.plot([0, 100], [0, 100], 'r--', label='Perfect Prediction') # Diagonal line
    plt.title('Fatigue Score: Actual vs Predicted')
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(config.BASE_DIR, "evaluation_plot.png")
    plt.savefig(plot_path)
    print(f"Saved evaluation plot to {plot_path}")

if __name__ == "__main__":
    evaluate_model()
