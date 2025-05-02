import random
import torch

class RandomCropWithMask:
    def __init__(self, size, shift):
        """
        Args:
            size (int or tuple): The desired crop size. If an int is provided, a square crop is used.
        """
        self.size = tuple(size)  # (crop_height, crop_width)
        self.shift = (size[0] - shift, size[1] - shift)

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image.
        
        Returns:
            cropped_img (PIL.Image): The randomly cropped image.
            mask (PIL.Image): A binary mask (same size as the original image) where the cropped region is 1 and the rest is 0.
        """
        w, h = img.size  # PIL Image gives size as (width, height)
        th, tw = self.shift  # crop height and width

        # Randomly select a crop position such that the crop fits in the image.
        top = random.randint(0, h - th)
        left = random.randint(0, w - tw)
        
        # Perform the crop using PIL's crop: box = (left, top, right, bottom)
        cropped_img = img.crop((left, top, left + tw, top + th))

        return cropped_img, (top, left, th, tw)
def sampling(H_img, W_img, output):
    """
    Generate a grid of coordinates for the given image dimensions.
    """
    DEVICE = 'cuda:0' # 'cuda' for fast computation
    output = output.to(DEVICE)
    y = torch.linspace(-1.0, 1.0, int(H_img), device=DEVICE)
    x = torch.linspace(-1.0, 1.0, int(W_img), device=DEVICE)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # [H_grid, W_grid]
    grid = torch.stack((xx, yy), dim=2).unsqueeze(0)
    grid = grid.to(DEVICE)
    # Reshape the grid to match the output shape
    sampling = torch.nn.functional.grid_sample(output, grid, align_corners=True)
    sampling.to(DEVICE)
    return sampling
# -------------------------

# -------------------------
def process_sample(image, keypoint_map, descriptor_map, crop_transform):
    cropped_image, (top, left, th, tw) = crop_transform(image)
    # Crop keypoint_map
    cropped_kp = keypoint_map[:,top:top+th, left:left+tw]

    # Crop descriptor_map [128,H/8,W/8]
    # Upsample the descriptor map to match the cropped image size
    # Note: Assuming descriptor_map is of shape [128, H/8, W/8]
    H_img, W_img = image.size
    W_cropped, H_cropped = cropped_image.size
    # print("cropped_image size:", cropped_image.size)
    desc_upsample = sampling(H_img, W_img, descriptor_map.unsqueeze(0))
   
    desc_upsample = desc_upsample[:,:,top:top+th, left:left+tw]
    cropped_desc = sampling(H_cropped/8, W_cropped/8, desc_upsample).squeeze(0)
   

    return cropped_image, cropped_kp, cropped_desc