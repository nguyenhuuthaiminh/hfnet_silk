import torch
import torch.nn.functional as F

def desc4silk(ret,out):
    """
    Upsample `local_descriptor_map` from [B, 128, 60, 80] to [B, 128, 480, 640]
    using `F.grid_sample()`.
    out: silk output including descriptor_position 
    """
    # Extract relevant tensors from the input dictionary
    keypoints = out['keypoints']  # shape: [N, 2]
    desc_map = ret['local_descriptor_map']  # [1, C, H', W']
    _, C, H_desc, W_desc = desc_map.shape

    # Extract original image shape to scale keypoints
    _, _, H_img, W_img = ret['image_shape']
    scale = torch.tensor([H_desc - 1, W_desc - 1], 
                         device=keypoints.device, 
                         dtype=torch.float32)
    scale /= torch.tensor([H_img - 1, W_img - 1], 
                        device=keypoints.device,
                        dtype=torch.float32)

    # Scale keypoints to descriptor map size
    keypoints_scaled = keypoints.float() * scale.unsqueeze(0).unsqueeze(0)
    
    # Normalize keypoints for grid_sample to [-1, 1]
    norm_kpts = keypoints_scaled.clone()
    norm_kpts[..., 0] = (norm_kpts[..., 0] / (H_desc - 1)) * 2 - 1
    norm_kpts[..., 1] = (norm_kpts[..., 1] / (W_desc - 1)) * 2 - 1
   
    # Sample descriptors
    grid = norm_kpts.unsqueeze(2)  # [1, N, 1, 2]
    local_desc = F.grid_sample(desc_map, grid, align_corners=True)  # [1, C, N, 1]
    local_desc = local_desc.squeeze(-1).permute(0, 2, 1)           # [1, N, C]
    local_desc = F.normalize(local_desc, p=2, dim=-1)

    ret['local_descriptor_map'] = local_desc

    return ret

if __name__ == '__main__':
    torch.manual_seed(42)

    # Example usage
    ret = {
        'dense_scores': torch.rand(1, 480, 640),        # shape: [B, H, W]
        'local_descriptor_map': torch.rand(1, 128, 60, 80),  # shape: [B, 128, H', W']
        'image_shape': torch.rand(1, 1, 480, 640).shape
    }
    out = {
        'keypoints': torch.randint(0,640,(100,2))  # shape: [100, 2]
    }
    print(out['keypoints'].shape)
    print(out['keypoints'][...,0])

    rets = desc4silk(ret,out)
    print(rets['local_descriptor_map'].shape)  # Expected: [1, 128, 480, 640]
