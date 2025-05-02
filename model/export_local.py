import torch
import torch.nn.functional as F
import numpy as np

from model.prediction import predict
from evaluation.utils.keypoints import keypoints_filter_borders, nms_fast

def normalize(l, axis=-1):
    return np.array(l) / np.linalg.norm(l, axis=axis, keepdims=True)


def root_descriptors(d, axis=-1):
    return np.sqrt(d / np.sum(d, axis=axis, keepdims=True))


def sample_bilinear(data, points):
    # Pad the input data with zeros
    data = np.pad(
        data, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=0)
    points = np.asarray(points) + 1

    x, y = points.T
    x0, y0 = points.T.astype(int)
    x1, y1 = x0 + 1, y0 + 1

    x0 = np.clip(x0, 0, data.shape[1]-1)
    x1 = np.clip(x1, 0, data.shape[1]-1)
    y0 = np.clip(y0, 0, data.shape[0]-1)
    y1 = np.clip(y1, 0, data.shape[0]-1)

    Ia = data[y0, x0]
    Ib = data[y1, x0]
    Ic = data[y0, x1]
    Id = data[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T

def sample_descriptors(descriptor_map, keypoints, image_shape,
                       input_shape=None, do_round=True):
    '''In some cases, the deep network computing the dense descriptors requires
       the input to be divisible by the downsampling rate, and crops the
       remaining pixels (PyTorch) or pad the input (Tensorflow). We assume the
       PyTorch behavior and round the factor between the network input
       (input shape) and the output (desc_shape). The keypoints are assumed to
       be at the scale of image_shape.
    '''
    fix = np.round if do_round else lambda x: x
    image_shape = np.array(image_shape)
    #print("image_shape: ", image_shape)
    desc_shape = np.array(descriptor_map.shape[:-1])
    #print("desc_shape: ", desc_shape)

    if input_shape is not None:
        input_shape = np.array(input_shape)
        factor = image_shape / input_shape
        effective_input_shape = desc_shape * fix(input_shape / desc_shape)
        factor = factor * (effective_input_shape - 1) / (desc_shape - 1)
    else:
        factor = (image_shape - 1) / (desc_shape - 1)
    desc = sample_bilinear(descriptor_map, keypoints/factor[::-1])
    #print("desc: ", desc)
    desc = normalize(desc, axis=1)
    assert np.all(np.isfinite(desc))
    return desc



def export_loader(ret, config, image):
    """
    Extracts keypoints, scores, and local descriptors from the model outputs with enhanced descriptor sampling.

    Parameters:
    - ret (dict): Dictionary containing:
        - "scores_dense" (torch.Tensor): Dense scores [H, W].
        - "local_descriptor_map" (torch.Tensor): Local descriptor map [H, W, C].
        - "logits" (torch.Tensor): Raw logits from the model [H, W, 65].
        - "prob_full" (torch.Tensor): Probabilities [H, W, 65].
        - "global_descriptor" (torch.Tensor): Global descriptor [D,].
        - "image_shape" (torch.tensor): Shape of the input image [C,H,W].
    - config (dict): Configuration containing:
        - "hfnet": Dictionary with keys:
            -"local": Dictionary with keys:
                - "detector_threshold" (float): Threshold for keypoint detection.
                - "nms_radius" (int): Radius for non-maximum suppression.
                - "num_keypoints" (int): Maximum number of keypoints to retain.
            -"do_nms" (bool): Whether to apply non-maximum suppression.
            -"nms_thresh": (float): Threshold for non-maximum suppression.
            -"remove_borders": (int): Border to remove.
    - input_shape (tensor): Shape of the input image [H, W].

    Returns:
    - dict: A dictionary with enhanced outputs.
    """
    
    pred = predict(ret, config)
    # print(f"Initial keypoints: {len(pred['keypoints'])}")
    

    keypoints = pred['keypoints'][0].squeeze(0).cpu().numpy()  # Convert back to NumPy
    scores = pred['scores'][0].detach().cpu().numpy()
    descriptors = pred['local_descriptors'][0].squeeze(0).detach().cpu().numpy()
    image_shape = image.shape[:2] # Get W, H from image shape
    # print(f"image shape: {image_shape}")

    # Apply border filtering if enabled
    remove_borders = config.get('remove_borders', 0)
    if remove_borders:
        # print(f"Removing borders: {remove_borders}")
        mask = keypoints_filter_borders(keypoints, image_shape, remove_borders)
        keypoints, scores, descriptors = keypoints[mask], scores[mask], descriptors[mask]
        # print(f"After border removal: {len(keypoints)}")

    # Apply NMS if enabled
    if config.get('do_nms', False):
        # print('applying NMS')
        keep = nms_fast(keypoints, scores, image_shape, config.get('nms_thresh', 4))
        keypoints, scores, descriptors = keypoints[keep], scores[keep], descriptors[keep]
        # print(f"After NMS: {len(keypoints)}")

    # Select top-K keypoints if needed
    num_features = config.get('num_features', 0)
    if num_features > 0:
        top_k = np.argsort(scores)[::-1][:num_features]
        keypoints, scores, descriptors = keypoints[top_k], scores[top_k], descriptors[top_k]
        # print(f"Selected top {num_features} keypoints")
        # print(f"Final selection: {len(keypoints)}")

    # # Optionally binarize descriptors
    if config.get('binarize', False):
        descriptors = descriptors > 0

    pred['keypoints'] = keypoints
    pred['scores'] = scores
    pred['local_descriptors'] = descriptors

    return pred

if "__main__" == __name__:

    # Example Usage
    config = {
        'local': {
            'detector_threshold': 0.005,
            'nms_radius': 4,
            'num_keypoints': 1000
        },
        'num_features': 300,
        'do_nms': True,
        'nms_thresh': 4,
        'remove_borders': 4,
    }
    

    # Mock outputs from the model
    ret = {
        'dense_scores': torch.rand(1, 480, 640),  # Example dense scores
        'local_descriptor_map': torch.rand(1, 256, 60, 80),  # Example descriptor map
        'logits': torch.rand(1,65,60, 80),  # Example logits
        'prob_full': torch.rand(1,65,60, 80),  # Example probabilities
        'global_descriptor': torch.rand(1,4096),  # Example global descriptor
        'image_shape': torch.tensor([1, 3, 480, 640])  # Example image shape
    }

    # ret = {
        # 'scores_dense': torch.rand(4, 480, 640),  # Example dense scores
        # 'local_descriptor_map': torch.rand(4, 256, 60, 80),  # Example descriptor map
        # 'image_shape': torch.tensor([4, 3, 480, 640])  # Example image shape
    # }

    # Example input image (just for scaling reference)
    image = torch.rand(3, 480, 640)  # Shape: [B,C, H, W]

    result = export_loader(ret, config, image)
    # Print output summary
    for key, value in result.items():
        print(f"{key}: {value.shape}")

