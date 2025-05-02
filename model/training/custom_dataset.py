import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

from model.training.utils import RandomCropWithMask

# --- Optional: Seed function (if not defined elsewhere) ---
# def setseed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# setseed(42)


# --- Custom Dataset ---
class CustomDataset(Dataset):
    def __init__(self, image_dir, glo_dir, loc_dir, num, size, shift,shift_transform=None, transform=None):
        self.image_dir = image_dir
        self.glo_dir = glo_dir
        self.loc_dir = loc_dir
        self.size = size
        self.shift = shift
        self.transform = transform
        self.shift_transform = shift_transform

        self.image_files = sorted(os.listdir(image_dir))[:num]
        self.glo_files = sorted(os.listdir(glo_dir))[:num]
        self.loc_files = sorted(os.listdir(loc_dir))[:num]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        glo_path = os.path.join(self.glo_dir, self.glo_files[idx])
        loc_path = os.path.join(self.loc_dir, self.loc_files[idx])

        glo_data = np.load(glo_path)
        loc_data = np.load(loc_path)

        glo = {k: torch.from_numpy(v) for k, v in glo_data.items()}
        loc = {k: torch.from_numpy(v) for k, v in loc_data.items()}

        # scores = loc['scores'].squeeze(-1)
        keypoint_map = loc['keypoint_map']  # shape: [1, H_d, W_d]
        # keypoint_map[keypoint_map != 0] = 1.0
        # keypoint_map = F.normalize(keypoint_map, p=2, dim=0)

        local_desc_map = loc['local_descriptor_map'] # shape: [128, H_d, W_d]

        local_desc_map = F.normalize(local_desc_map, p=2, dim=0)

        crop_transform = RandomCropWithMask(self.size,self.shift)
        image = Image.open(img_path).convert('L')
        image = image.resize(self.size[::-1])
        cropped_img, mask = crop_transform(image)
        
        if self.shift_transform:
            image,  keypoint_map, local_desc_map = self.shift_transform(image, keypoint_map, local_desc_map,crop_transform)
        
        if self.transform:
            image = self.transform(image)  # shape: [C, H, W]
        
        inp = {
            "global_descriptor": glo['global_descriptor'],
            "keypoint_map": keypoint_map,
            "local_descriptor_map": local_desc_map,
        }

        return image, inp
