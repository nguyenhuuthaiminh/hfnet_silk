from torch import nn
import torch
import torch.nn.functional as F
import torch.nn.init as init


class VLAD(nn.Module):
    def __init__(self, config):
        super(VLAD, self).__init__()
        self.intermediate_proj = config.get('intermediate_proj', None)
        if self.intermediate_proj:
            self.pre_proj = nn.Conv2d(240, self.intermediate_proj, kernel_size=1)

        self.n_clusters = config['n_clusters']
        self.memberships = nn.Conv2d(240, self.n_clusters, kernel_size=1)

        # Cluster centers
        self.clusters = nn.Parameter(torch.empty(1, self.n_clusters, 240))
        self._initialize_weights()  # Initialize the cluster weights

    def _initialize_weights(self):
        # Xavier initialization for cluster weights
        init.xavier_uniform_(self.clusters)

    def forward(self, feature_map, mask=None):
        if self.intermediate_proj:
            feature_map = self.pre_proj(feature_map)

        batch_size, _, h, w = feature_map.size()

        # Compute memberships (soft-assignment)
        memberships = F.softmax(self.memberships(feature_map), dim=1)

        # Reshape feature_map and clusters for broadcasting
        feature_map = feature_map.permute(0, 2, 3, 1).unsqueeze(3)  # (B, H, W, 1, D)
        residuals = self.clusters - feature_map  # Compute residuals
        residuals = residuals * memberships.permute(0, 2, 3, 1).unsqueeze(4)  # Weight residuals by memberships

        if mask is not None:
            residuals = residuals * mask.unsqueeze(-1).unsqueeze(-1)

        # Sum residuals to form the VLAD descriptor
        descriptor = residuals.sum(dim=[1, 2])

        # Intra-normalization
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        # Flatten descriptor and apply L2 normalization
        descriptor = descriptor.view(batch_size, -1)
        descriptor = F.normalize(descriptor, p=2, dim=1)

        return descriptor

class DimensionalityReduction(nn.Module):
    def __init__(self, config, proj_regularizer=None):
        """
        Initializes the Dimensionality Reduction module.

        Args:
            input_dim (int): Dimension of the input feature descriptor.
            output_dim (int): Dimension of the reduced descriptor.
            proj_regularizer (float, optional): L2 regularization strength. If None, no regularization is applied.
        """
        super(DimensionalityReduction, self).__init__()
        input_dim = config['n_clusters'] * 240
        output_dim = config['dimensionality_reduction']
        self.proj_regularizer = proj_regularizer

        # Fully connected layer with Xavier initialization
        self.fc = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.fc.weight)

        # Optional L2 regularization
        if proj_regularizer is not None:
            self.regularizer = lambda w: proj_regularizer * torch.sum(w ** 2)
        else:
            self.regularizer = None

    def forward(self, descriptor):
        """
        Forward pass for the Dimensionality Reduction module.

        Args:
            descriptor (torch.Tensor): Input feature descriptor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reduced and normalized descriptor of shape (batch_size, output_dim).
        """
        # Normalize the input descriptor
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        # Apply the fully connected layer
        descriptor = self.fc(descriptor)

        # Normalize the output descriptor
        descriptor = F.normalize(descriptor, p=2, dim=-1)

        # Apply regularization if specified
        if self.regularizer is not None:
            reg_loss = self.regularizer(self.fc.weight)
            return descriptor, reg_loss

        return descriptor

class LocalHead(nn.Module):
    def __init__(self, config):
        super(LocalHead, self).__init__()
        descriptor_dim = config['descriptor_dim']
        detector_grid = config['detector_grid']

        # Descriptor Head
        self.desc_conv1 = nn.Conv2d(config['input_channels'], descriptor_dim, kernel_size=3, stride=1, padding=1)
        self.desc_bn1 = nn.BatchNorm2d(descriptor_dim)
        self.desc_conv2 = nn.Conv2d(descriptor_dim, descriptor_dim, kernel_size=1, stride=1, padding=0)

        # Detector Head
        self.det_conv1 = nn.Conv2d(config['input_channels'], 128, kernel_size=3, stride=1, padding=1)
        self.det_bn1 = nn.BatchNorm2d(128)
        self.det_conv2 = nn.Conv2d(128, detector_grid ** 2, kernel_size=1, stride=1, padding=0)

        self.detector_grid = detector_grid

    def forward(self, features):
        # Descriptor Head
        desc = F.relu6(self.desc_bn1(self.desc_conv1(features)))
        desc = self.desc_conv2(desc)
        desc = F.normalize(desc, p=2, dim=1)

        # Detector Head
        logits = F.relu6(self.det_bn1(self.det_conv1(features)))
        logits = self.det_conv2(logits)
        logits = F.pixel_shuffle(logits, self.detector_grid)
        
        prob = F.sigmoid(logits)

        return desc, logits, prob