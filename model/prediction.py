import torch
import torch.nn.functional as F

def simple_nms(scores, radius, iterations=3):
    """
    Performs non-maximum suppression (NMS) on a 2D heatmap using max-pooling.

    Args:
        scores (torch.Tensor): 2D tensor of shape [H, W] representing the heatmap.
        radius (int): Neighborhood radius for local maxima.
        iterations (int): Number of NMS iterations to apply.

    Returns:
        torch.Tensor: Heatmap with non-maxima suppressed to zero.
    """
    size = 2 * radius + 1

    def max_pool(x):
        # Temporarily add batch & channel dims for max_pool2d
        return F.max_pool2d(x.unsqueeze(0).unsqueeze(0), kernel_size=size, stride=1, padding=radius).squeeze()

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)

    for _ in range(iterations - 1):
        suppression_mask = max_pool(max_mask.float()).bool()
        suppressed_scores = torch.where(suppression_mask, zeros, scores)
        new_max_mask = suppressed_scores == max_pool(suppressed_scores)
        max_mask |= new_max_mask & ~suppression_mask

    return torch.where(max_mask, scores, zeros)


def predict(ret, config):
    """
    Extracts keypoints and local descriptors from model outputs. Batch size must be 1.

    Args:
        ret (dict): Dictionary containing:
            - 'dense_scores': [1, 1, H, W] score maps
            - 'local_descriptor_map': [1, C, H', W'] descriptor maps
            - 'image_shape': shape info for scaling keypoints
        config (dict): Configuration dict with 'local' sub-dict specifying
                       nms_radius, detector_threshold, num_keypoints, etc.

    Returns:
        dict: Updated 'ret' with new keys:
              'keypoints', 'scores', 'local_descriptors'
    """
    dense_scores = ret['dense_scores'].squeeze(0).squeeze(0) # shape: [1, H, W]

    scores = dense_scores
    if config['local'].get('nms_radius', 0) > 0:
        scores = simple_nms(scores, config['local']['nms_radius'])  

    # Threshold keypoints
    mask = scores >= config['local']['detector_threshold']
    keypoints = torch.nonzero(mask, as_tuple=False)  # [N, 2]
    kp_scores = scores[keypoints[:, 0], keypoints[:, 1]]

    # Keep top-k if specified
    k = config['local'].get('num_keypoints', None)
    if k is not None and k > 0:
        k = min(keypoints.shape[0], k)
        kp_scores, idx = torch.topk(kp_scores, k)
        keypoints = keypoints[idx] # [x,y]

    # Prepare descriptor map
    desc_map = ret['local_descriptor_map']  # [1, C, H', W']
    _, C, H_desc, W_desc = desc_map.shape

    # Extract original image shape to scale keypoints
    _, _, H_img, W_img = ret['image_shape']
    scale = torch.tensor([H_desc - 1,W_desc - 1], 
                         device=scores.device, 
                         dtype=torch.float32)
    scale /= torch.tensor([H_img - 1,W_img - 1], 
                          device=scores.device, 
                          dtype=torch.float32)

    # Scale keypoints to descriptor map size
    keypoints_scaled = keypoints.float() * scale.unsqueeze(0).unsqueeze(0)

    # Normalize keypoints for grid_sample to [-1, 1]
    norm_kpts = keypoints_scaled.clone()
    norm_kpts[..., 0] = (norm_kpts[..., 0] / (H_desc - 1)) * 2 - 1
    norm_kpts[..., 1] = (norm_kpts[..., 1] / (W_desc - 1)) * 2 - 1

    # Sample descriptors
    grid = norm_kpts.unsqueeze(2)  # [1, N, 1, 2]
    local_desc = F.grid_sample(desc_map, grid, align_corners=False)  # [1, C, N, 1]
    local_desc = local_desc.squeeze(-1).permute(0, 2, 1)           # [1, N, C]
    local_desc = F.normalize(local_desc, p=2, dim=-1)

    # Stack results for the batch
    device = dense_scores.device
    ret.update({
        'keypoints': keypoints.unsqueeze(0).to(device),
        'scores': kp_scores.unsqueeze(0).to(device),
        'local_descriptors': local_desc.to(device)
    })
    
    return ret


if __name__ == '__main__':
    ret = {
        'local_descriptor_map': torch.rand(1, 128, 60, 80),
        'dense_scores': torch.rand(1, 1,480,640),
        'image_shape': torch.tensor([1, 3, 480, 640])
    }

    config = {
        'local': {
            'detector_threshold': 0.001,
            'nms_radius': 4,
            'num_keypoints': 800
        }
    }

    ret = predict(ret, config)
    for k, v in ret.items():
        print(k, v.shape)
        if k == 'keypoints':
            print('keypoints')
            # show the maximum keypoint positions
            print(max(v[0, :, 0]), max(v[0, :, 1]))