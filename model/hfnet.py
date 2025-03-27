import torch
import torch.nn as nn
import torch.nn.functional as F
import math


from model.utils.layer import VLAD, DimensionalityReduction, LocalHead
from model.losses import descriptor_global_loss, descriptor_local_loss, detector_loss

def image_normalization(image, pixel_value_offset=128.0, pixel_value_scale=128.0):
    return (image - pixel_value_offset) / pixel_value_scale

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class HFNet(nn.Module):
    def __init__(self, config, width_mult=1.0):
        super(HFNet, self).__init__()
        # [expand_ratio, channels, repeats, stride]
        self.cfgs = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 1, 2],
            [6,  64, 1, 1],
            [6, 128, 1, 1], # Brach here
            [6, 64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        image_channels = config['image_channels']
        # Feature Extractor (MobileNetV2 backbone)
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(image_channels, input_channel, 2)]
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # Keypoint Detector Head
        self.local_head = LocalHead(config['local_head'])

        # Descriptor Head
        self.global_head = nn.Sequential(
            VLAD(config['global_head']),
            DimensionalityReduction(config['global_head'])
        )
        self.logvars = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], requires_grad=True))
        self._initialize_weights()

    def forward(self, x):
        x = image_normalization(x)
        # Backbone
        features_1 = self.features[:7](x)
        features_2 = self.features[7:](features_1)

        # local_head
        desc, logits, prob = self.local_head(features_1)

        # Classification (if needed)
        descriptor = self.global_head(features_2)

        return {'local_descriptor_map':desc,
                'logits':logits,
                'dense_scores':prob,
                'global_descriptor':descriptor,
                'image_shape':x.shape
                }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
    def _compute_loss(self, inputs, outputs, config):
        """Computes the total loss using external loss functions."""
        desc_g = descriptor_global_loss(inputs, outputs).mean()
        desc_l = descriptor_local_loss(inputs, outputs).mean()
        detect = detector_loss(inputs, outputs, config).mean()

        # Apply weighting
        if config['loss_weights'] == 'uncertainties':
            w = {f'logvar_{i}': logvar.item() for i, logvar in enumerate(self.logvars)}
            precisions = [torch.exp(-logvar) for logvar in self.logvars]
            total_loss = desc_g * precisions[0] + (self.logvars[0])
            total_loss += desc_l * precisions[1] + (self.logvars[1])
            total_loss += 2 * detect * precisions[2] + (self.logvars[2])
        else:
            w = config['loss_weights']
            total_loss = (
                (w['global'] * desc_g + w['local'] * desc_l + w['detector'] * detect) /
                sum(w.values())
            )
        

        return total_loss, {
            'global_desc_l2': desc_g.item(),
            'local_desc_l2': desc_l.item(),
            'detector_crossentropy': detect.item()
        }, w
    

if __name__ == '__main__':
    image = torch.randn(1, 1, 480, 640)

    config= {
    'image_channels':1,
    'loss_weights': 'uncertainties',
    'local':{
        'detector_threshold': 0.001,
        'nms_radius': 4,
        'num_keypoints': 10000
    },
    'local_head': {
        'descriptor_dim': 128,
        'detector_grid': 8,
        'input_channels': 96
    },
    'global_head': {
        'n_clusters': 32,
        'intermediate_proj': 0,
        'dimensionality_reduction': 4096
        }
    }
    model = HFNet(config, width_mult=0.75)

    ret = model(image)
    for k, v in ret.items():
        if k == 'scores_dense':
            print(k, v.shape)
        elif k == 'local_descriptor_map':
            print(k, v.shape)

    