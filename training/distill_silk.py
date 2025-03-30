import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, image_dir, glo_dir, loc_dir, num, num_keypoints, transform=None):
        self.image_dir = image_dir
        self.glo_dir = glo_dir
        self.loc_dir = loc_dir
        self.num_keypoints = num_keypoints
        self.image_files = sorted(os.listdir(image_dir))[:num]
        self.glo_files = sorted(os.listdir(glo_dir))[:num]
        self.loc_files = sorted(os.listdir(loc_dir))[:num]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        glo_path = os.path.join(self.glo_dir, self.glo_files[idx])
        loc_path = os.path.join(self.loc_dir, self.loc_files[idx])
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load global and local descriptors
        glo = {key: torch.from_numpy(value) for key, value in np.load(glo_path).items()}
        loc = {key: torch.from_numpy(value) for key, value in np.load(loc_path).items()}
        
        scores = loc['scores'].squeeze(-1)
        keypoints = loc['keypoint']
        local_descriptors = loc['local_descriptor_map']

        # Select top-k keypoints
        topk_indices = scores.topk(self.num_keypoints, largest=True).indices
        keypoints = keypoints[topk_indices]
        keypoints = keypoints.long()
        scores = scores[topk_indices]
        local_descriptors = local_descriptors[topk_indices]
                
        _, H_desc, W_desc = image.shape
        local_descriptor_map = torch.zeros(128, H_desc, W_desc)
        keypoint_map = torch.zeros(1,H_desc, W_desc)
        local_descriptor_map[:, keypoints[:, 0], keypoints[:, 1]] = local_descriptors.T
        
        keypoint_map[:, keypoints[:, 0], keypoints[:, 1]] = 1

        inp = {
            "global_descriptor": glo['global_descriptor'],  
            "keypoint_map": keypoint_map,
            "local_descriptor_map": local_descriptor_map,
        }

        return image, inp
