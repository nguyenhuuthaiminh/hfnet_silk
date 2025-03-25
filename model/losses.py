import torch
import torch.nn.functional as F

def image_normalization(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    """
    Normalizes an image tensor by subtracting an offset and dividing by a scale.
    Typically used to convert [0,255] RGB values into a normalized range.
    
    Args:
        image (torch.Tensor): Input image of shape [C, H, W] or [B, C, H, W].
        pixel_value_offset (float): Value to subtract from each pixel.
        pixel_value_scale (float): Value to divide each pixel after offset.
    
    Returns:
        torch.Tensor: Normalized image tensor.
    """
    return (image - pixel_value_offset) / pixel_value_scale

def descriptor_global_loss(inp, out):
    """
    Computes the L2 loss between the input and output global descriptors.
    
    Args:
        inp (dict): Dictionary containing 'global_descriptor' tensor of shape [B, D].
        out (dict): Dictionary containing 'global_descriptor' tensor of shape [B, D].
    
    Returns:
        torch.Tensor: Scalar L2 loss for each sample in the batch, shape [B].
    """
    d = torch.square((inp['global_descriptor'] - out['global_descriptor']))
    return torch.sum(d, dim=-1) / 2

def descriptor_local_loss(inp, out):
    """
    Computes the L2 loss for local descriptors, which are spatial feature maps.
    
    Args:
        inp (dict): Contains 'local_descriptor_map' of shape [B, C, H, W].
        out (dict): Contains 'local_descriptor_map' of shape [B, C, H, W].
    
    Returns:
        torch.Tensor: L2 loss map of shape [B, H, W].
                     Each pixel's loss is averaged across channels.
    """
    d = torch.square(inp['local_descriptor_map'] - out['local_descriptor_map'])
    # Sum over channels and divide by 2
    d = torch.sum(d, dim=1) / 2
    return d

def detector_loss(inp, out, config, threshold=0.5):
    """
    Computes cross-entropy loss for keypoints or dense scores.
    
    Args:
        inp (dict): Must contain either 'keypoint_map' or 'dense_scores'.
                    - 'keypoint_map' should be [B, 1, H, W] or [B, C, H, W].
                    - 'dense_scores' should be [B, C, H, W].
        out (dict): Contains 'logits' of shape [B, C, H, W].
        config (dict): Configuration dictionary with 'local_head' subkey 'detector_grid'.
        threshold (float): Not currently used directly in the function, but
                           included for future modifications (e.g., if you need
                           to threshold logits or labels).
    
    Returns:
        torch.Tensor: Scalar cross-entropy loss.
    """
    logits = out['logits']  # [B, C, H, W]
    B, _, H, W = logits.shape
    grid_size = config['local_head']['detector_grid']

    if 'keypoint_map' in inp: # Hard Labels
        labels = inp['keypoint_map']  # e.g. [B, 1, H, W]
        # Pixel unshuffle
        labels = F.pixel_unshuffle(labels, grid_size)  # [B, grid_size^2, H', W']
        
        # Concatenate channel of ones for background vs. foreground classes
        # Note: This effectively creates 2 channels: 2*labels vs. 1
        # Then argmax picks whichever channel is higher.
        ones_channel = torch.ones(B, 1, H, W, device=logits.device)
        labels = torch.cat((2 * labels, ones_channel), dim=1)
        labels = torch.argmax(labels, dim=1)  # [B, H, W]
        
        # Cross entropy expects class indices in [0, C-1]
        loss = F.cross_entropy(logits, labels.long())

    elif 'dense_scores' in inp: # Soft labes
        # If dense_scores is used as a multi-class probability map or similar
        labels = inp['dense_scores']  # [B, C, H, W]
        
        loss = F.cross_entropy(logits, labels)

    else:
        raise ValueError("Input must contain 'keypoint_map' or 'dense_scores'.")

    return loss
