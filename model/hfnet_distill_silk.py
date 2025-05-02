from model.hfnet import HFNet
from model.training.train import train_model

from model.training.custom_dataset import CustomDataset

import torch

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from model.training.config import IMAGE_DIR, GLOBAL_DIR, LOCAL_DIR
from model.training.utils import process_sample

if __name__ == '__main__':
    print("🔥 Script started")

    # --- Paths ---
    dataset_path = IMAGE_DIR
    global_path = GLOBAL_DIR
    local_path = LOCAL_DIR

    # --- Dataset Initialization ---
    # Define Transformations

    transform = transforms.Compose([ 
        transforms.Resize((480, 640)),  # Resize to (480, 640)
        transforms.ColorJitter(
            brightness=0.2,  # Adjust based on delta_range [-30, 40]
            contrast=0.5     # Adjust based on strength_range [0.3, 1.2]
        ),
        transforms.ToTensor()
    ])

    # Dataset Initialization
    data_set = CustomDataset(
        image_dir=dataset_path, 
        glo_dir=global_path, 
        loc_dir=local_path,  
        num=1000,  # Number of images, 
        size = (480,640),
        shift = 8,
        shift_transform = None,
        transform=transform,

    )

    # Train, Validation, Test Split
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.2
    BATCH_SIZE = 1

    train_size = int(TRAIN_SIZE * len(data_set))
    val_size = int(VAL_SIZE * len(data_set))
    test_size = len(data_set) - train_size - val_size

    torch.manual_seed(42)  # Ensure reproducibility
    train_dataset, val_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])
    # ⚡ Optimized DataLoaders ⚡
    num_workers = 2  # Use more workers for faster data loading

    # ✅ Create DataLoader with Deterministic Shuffle
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle= True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)


    config= {
        'image_channels':1,
        # 'loss_weights': 'uncertainties',
        'loss_weights':{
            'global':1,
            'local':1,
            'detector':1
        },
        'local_head': {
            'descriptor_dim': 128,
            'detector_grid': 8,
            'input_channels': 96
        },
        'global_head': {
            'n_clusters': 32,
            'intermediate_proj': 0,
            'dimensionality_reduction': 4096
        }
    }
    model = HFNet(config, width_mult=0.75)
    train_model(model, train_loader, val_loader, config, lr=2e-4, patience = 100,milestone=[101], epochs=100)