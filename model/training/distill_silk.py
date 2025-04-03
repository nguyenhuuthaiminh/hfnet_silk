import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A dataset class that:
      - Loads an image from 'image_dir'
      - Loads corresponding global and local descriptors from 'glo_dir' and 'loc_dir'
      - Selects top-k keypoints by score
      - Constructs descriptor and keypoint maps at the image resolution
    """
    def __init__(
        self,
        image_dir: str,
        glo_dir: str,
        loc_dir: str,
        num: int,
        num_keypoints: int,
        transform=None
    ):
        """
        Args:
            image_dir (str): Directory containing images.
            glo_dir (str): Directory containing global descriptors (.npz files).
            loc_dir (str): Directory containing local descriptors/keypoints (.npz files).
            num (int): Number of samples to load.
            num_keypoints (int): Number of top keypoints to keep based on scores.
            transform (callable, optional): Transform to be applied on the PIL image.
        """
        self.image_dir = image_dir
        self.glo_dir = glo_dir
        self.loc_dir = loc_dir
        self.num_keypoints = num_keypoints
        self.transform = transform

        # Collect file names, sorted for consistent ordering
        self.image_files = sorted(os.listdir(image_dir))[:num]
        self.glo_files = sorted(os.listdir(glo_dir))[:num]
        self.loc_files = sorted(os.listdir(loc_dir))[:num]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): The transformed image of shape [C, H, W].
            inp (dict): Dictionary containing:
                "global_descriptor" (torch.Tensor),
                "keypoint_map" (torch.Tensor of shape [1, H, W]),
                "local_descriptor_map" (torch.Tensor of shape [128, H, W])
        """
        # File paths
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        glo_path = os.path.join(self.glo_dir, self.glo_files[idx])
        loc_path = os.path.join(self.loc_dir, self.loc_files[idx])
        
        # Load and transform the image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # shape: [C, H, W]

        # Load descriptors
        glo_data = np.load(glo_path)
        loc_data = np.load(loc_path)

        glo = {k: torch.from_numpy(v) for k, v in glo_data.items()}
        loc = {k: torch.from_numpy(v) for k, v in loc_data.items()}

        # Extract local data
        scores = loc['scores'].squeeze(-1)         # shape: [N]
        keypoints = loc['keypoint']               # shape: [N, 2]
        local_descriptors = loc['local_descriptor_map']  # shape: [N, 128]
        

        # Select top-k keypoints
        topk_indices = scores.topk(self.num_keypoints, largest=True).indices
        keypoints = keypoints[topk_indices].long()
        scores = scores[topk_indices]
        local_descriptors = local_descriptors[topk_indices]
        local_descriptors = F.normalize(local_descriptors, p=2, dim =-1)

        # Build descriptor & keypoint maps
        _, H_img, W_img = image.shape  # shape: [C, H, W]
        keypoint_map = torch.zeros(1, H_img, W_img)
        
        # Place scores in the keypoint map
        keypoint_map[:, keypoints[:, 0], keypoints[:, 1]] = scores
        # keypoint_map = F.normalize(keypoint_map, p=2, dim =0)

        # Build dictionary
        inp = {
            "global_descriptor": glo['global_descriptor'],
            "keypoint_map": keypoint_map,
            "local_descriptor_map": local_descriptors,
            "keypoints": keypoints
        }

        return image, inp
