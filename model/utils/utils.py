import torch
import torch.nn.functional as F

def desc4silk(ret):
    """
    Upsample `local_descriptor_map` from [B, 128, 60, 80] to [B, 128, 480, 640]
    using `F.grid_sample()`.
    """
    # Extract relevant tensors from the input dictionary
    dense_scores = ret['dense_scores']  # shape: [B, H, W]
    local_descriptor_map = ret['local_descriptor_map']  # shape: [B, 128, H', W']

    B, H, W = dense_scores.shape
    _, C, H_ld, W_ld = local_descriptor_map.shape  # For example, 128, 60, 80

    # 1) Build a normalized sampling grid in the range [-1, 1].
    #    H, W here are the desired output dimensions (480, 640).
    y = torch.linspace(-1, 1, H, device=local_descriptor_map.device)
    x = torch.linspace(-1, 1, W, device=local_descriptor_map.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')  # shape: [H, W] each

    # 2) Combine x, y into a single grid of shape [H, W, 2], then expand to [B, H, W, 2].
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(B, -1, -1, -1)

    # 3) Upsample the descriptor map using bilinear interpolation.
    #    `align_corners=True` ensures corners in [-1,1] align with corner pixels.
    upsampled_map = F.grid_sample(
        local_descriptor_map,
        grid,
        mode='bilinear',
        align_corners=True
    )

    # Update the dictionary with the upsampled map
    ret['local_descriptor_map'] = upsampled_map
    return ret

if __name__ == '__main__':
    torch.manual_seed(42)

    # Example usage
    ret = {
        'dense_scores': torch.rand(1, 480, 640),        # shape: [B, H, W]
        'local_descriptor_map': torch.rand(1, 128, 60, 80),  # shape: [B, 128, H', W']
        'image_shape': torch.rand(1, 1, 480, 640)
    }

    rets = desc4silk(ret)
    print(rets['local_descriptor_map'].shape)  # Expected: [1, 128, 480, 640]
