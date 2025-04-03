#!/home/tupv8/miniconda3/envs/reproduce/bin/python


from hfnet import HFNet
from training.train import train_model

from training.distill_silk import CustomDataset

import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from training.config import IMAGE_DIR, GLOBAL_DIR, LOCAL_DIR

def setseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == "__main__":
    setseed(42)

    # --- Transforms ---
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.5
        ),
        transforms.ToTensor(),
    ])

    # --- Paths ---
    dataset_path = IMAGE_DIR
    global_path = GLOBAL_DIR
    local_path = LOCAL_DIR

    # --- Dataset Initialization ---
    data_set = CustomDataset(
        image_dir=dataset_path,
        glo_dir=global_path,
        loc_dir=local_path,
        num=10,          # Number of images to load
        num_keypoints=10000,
        transform=transform
    )

    # --- Train/Val/Test Split ---
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    BATCH_SIZE = 1

    train_size = int(TRAIN_RATIO * len(data_set))
    val_size = int(VAL_RATIO * len(data_set))
    test_size = len(data_set) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        data_set,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # --- Dataloaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,       # Helpful if using a GPU
        num_workers=2          # Adjust based on your system
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=2
    )

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
    train_model(model, train_loader, val_loader, config, lr=1e-3, patience = 150,milestone=[150], epochs=150)