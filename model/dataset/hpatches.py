import torch
import torch.utils.data as data
import cv2
import numpy as np
from pathlib import Path

DATA_PATH = '/media/tupv8/32f7b466-e4d4-4f6f-b2aa-1e5a8ce7663c/vinai/hfnet'
from torch.utils.data import DataLoader


class HpatchesDataset(data.Dataset):
    def __init__(self, alteration='all', truncate=None, make_pairs=False, hard=False,
                 shuffle=False, random_seed=0, preprocessing={'resize_max': 640}):
        self.dataset_folder = 'hpatches'
        self.num_images = 6
        self.image_ext = '.ppm'
        self.make_pairs = make_pairs
        self.preprocessing = preprocessing
        
        base_path = Path(DATA_PATH, self.dataset_folder)
        scene_paths = sorted([x for x in base_path.iterdir() if x.is_dir()])
        
        self.data = {'image': [], 'name': []}
        if make_pairs:
            self.data.update({'image2': [], 'name2': [], 'homography': []})
        
        for path in scene_paths:
            if alteration == 'i' and path.stem[0] != 'i':
                continue
            if alteration == 'v' and path.stem[0] != 'v':
                continue
            for i in range(1, 1 + self.num_images):
                if make_pairs:
                    if i == 1:
                        path2 = str(Path(path, '1' + self.image_ext))
                        name2 = path.stem + '/1'
                        continue
                    if hard and i < self.num_images:
                        continue
                    self.data['image2'].append(path2)
                    self.data['name2'].append(name2)
                    self.data['homography'].append(
                        np.loadtxt(str(Path(path, 'H_1_' + str(i)))))
                self.data['image'].append(str(Path(path, str(i) + self.image_ext)))
                self.data['name'].append(path.stem + '/' + str(i))
        
        if shuffle:
            rng = np.random.RandomState(random_seed)
            perm = rng.permutation(len(self.data['name']))
            self.data = {k: [v[i] for i in perm] for k, v in self.data.items()}
        if truncate:
            self.data = {k: v[:truncate] for k, v in self.data.items()}
    
    def __len__(self):
        return len(self.data['name'])
    
    def __getitem__(self, idx):
        image = cv2.imread(self.data['image'][idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_size = image.shape[:2]
        
        if self.preprocessing.get('resize_max'):
            image = self._resize_max(image, self.preprocessing['resize_max'])
        
        sample = {'image': torch.tensor(image, dtype=torch.float32),
                  'original_size': original_size,
                  'name': self.data['name'][idx]}
        
        if self.make_pairs:
            image2 = cv2.imread(self.data['image2'][idx])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            original_size2 = image2.shape[:2]
            
            if self.preprocessing.get('resize_max'):
                image2 = self._resize_max(image2, self.preprocessing['resize_max'])
            
            H_raw = torch.tensor(self.data['homography'][idx], dtype=torch.float32).view(3, 3)

            # Apply homography adaptation after resizing
            sample.update({'image2': torch.tensor(image2, dtype=torch.float32),
                           'original_size2': original_size2,
                           'name2': self.data['name2'][idx]})

            sample['homography'] = adapt_homography_to_preprocessing(H_raw, sample)
        
        return sample
    
    def _resize_max(self, image, resize_max):
        h, w = image.shape[:2]
        scale = resize_max / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)


def adapt_homography_to_preprocessing(H, data):
    """ Adjusts homography matrix according to image resizing. """
    size1 = torch.tensor(data['image'].shape[-2:], dtype=torch.float32)  # (H, W)
    size2 = torch.tensor(data['image2'].shape[-2:], dtype=torch.float32)  # (H, W)

    s1 = size1 / torch.tensor(data['original_size'], dtype=torch.float32)
    s2 = size2 / torch.tensor(data['original_size2'], dtype=torch.float32)

    mult1 = torch.diag(torch.cat([s1, torch.tensor([1.0])]))
    mult2 = torch.diag(torch.cat([1 / s2, torch.tensor([1.0])]))

    return mult1 @ H @ mult2  # Equivalent to tf.matmul(mult1, tf.matmul(H, mult2))


# Usage Example
if __name__ == "__main__":
    # Load dataset
    dataset = HpatchesDataset(alteration='all', make_pairs=True, shuffle=True)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get one batch
    for batch in dataloader:
        print("\n--- Batch Information ---")

        # Check shape of images
        print(f"Image tensor shape: {batch['image'].shape}")  # (B, 1, H, W)

        if 'image2' in batch:
            print(f"Image2 tensor shape: {batch['image2'].shape}")  # (B, 1, H, W)

        print(f"Original sizes: {batch['original_size']}")  
        if 'original_size2' in batch:
            print(f"Original sizes (image2): {batch['original_size2']}")  

        print(f"Names: {batch['name']}")
        if 'name2' in batch:
            print(f"Names (image2): {batch['name2']}")

        if 'homography' in batch:
            print(f"Homography shape: {batch['homography'].squeeze(0).shape}")  # Should be (B, 3, 3)

        break
