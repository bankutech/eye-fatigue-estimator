import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyre-ignore[21]
import config

class FatigueDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data_frame)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        
        # Get the fatigue score and convert to float32 tensor
        score = float(self.data_frame.iloc[index, 1])
        score = torch.tensor([score], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, score

def get_transforms(is_train=True):
    """
    Returns torchvision transforms for training and validation.
    MobileNetV2 expects normalization with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
