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
        
        # scores = loc['dense_scores']
        keypoints = loc['keypoints']
        local_descriptors = loc['local_descriptors']
        # generate local_descriptor_map
        _, H_desc, W_desc = image.shape
        local_descriptor_map = torch.zeros(1, 128, H_desc, W_desc)
        for i, (x, y) in enumerate(keypoints):
            local_descriptor_map[0, :, y, x] = local_descriptors[i]
        

        # Select top-k keypoints
        topk_indices = scores.topk(self.num_keypoints, largest=True, sorted=True).indices
        keypoints = keypoints[topk_indices]
        scores = scores[topk_indices]
        local_descriptors = local_descriptors[topk_indices]

        inp = {
            "global_descriptor": glo['global_descriptor'],  
            "keypoints": keypoints,
            # "dense_scores": scores.permute(2,0,1),
            "local_descriptor_map": local_descriptor_map,
        }

        return image, inp
