import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F


class RCNN(nn.Module):
    def __init__(self, model, n_classes, pool_size=7, pretrained=True):
        super().__init__()
        self.pool_size = pool_size
        if pretrained:
            mod = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        else:
            mod = torchvision.models.vgg16(weights=None)
            self._layers = nn.Sequential(*[mod.classifier[i] for i in [0,1,3,4]])
        self._delta = nn.Linear(4096, 4*(n_classes+1))
        self._target = nn.Linear(4096, n_classes+1)
    
    def _post_process(self, onehot_delta, targets):
        max_values, max_idx = targets.max(dim=1)
        delta = torch.stack(
            [bx[idx*4:idx*4+4] for bx, idx in zip(onehot_delta, max_idx)],
            dim=0
        )
        return delta

    def forward(self, features: torch.Tensor, roi: torch.Tensor):
        """ Forward pass of Region Based CNN

        Args:
            features (torch.Tensor): extracted features (B, C, H, W)
            roi (torch.Tensor): regoin of interest(s) (N, 4)
        
        Returns:
            torch.Tensor: bounding deltas (N, 4)
            torch.Tensor: predicted targets (N, n_channels+1)
        """
        
        pooled_features = torchvision.ops.roi_pool(
            features.float(), 
            [roi.float()], 
            output_size=(self.pool_size, self.pool_size)
        )
        pooled_features = pooled_features.view(pooled_features.shape[0], -1)
        x = self._layers(pooled_features)
        onehot_delta = self._delta(x)
        targets = F.softmax(self._target(x), dim=1)
        delta = self._post_process(onehot_delta, targets)
        return delta, targets
