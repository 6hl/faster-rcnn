import torch
import torch.nn as nn
from torch.nn import functional as F


class RegionProposalNetwork(nn.Module):
    """ Class used to map extracted features to image space using anchors """
    def __init__(self, n_classes):
        super().__init__()
        # TODO: Init weights?
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_anchors = 9
        self._conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding="same")
        self._target = nn.Conv2d(in_channels=512, out_channels=n_anchors*2, kernel_size=(1,1), stride=1, padding=0)
        self._delta = nn.Conv2d(in_channels=512, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, features):
        """ Forward pass of RPN

        Args:
            features (torch.Tensor): extracted features
        
        Returns:
            torch.Tensor: bounding deltas (b, N, 4)
            torch.Tensor: predicted targets fg/bg (b, N, 2)
        """
        x = F.relu(self._conv(features))
        target = self._target(x).permute(0, 2, 3, 1).contiguous()
        target = F.softmax(target.view(features.shape[0], -1, 2), dim=2)
        delta = self._delta(x).permute(0,2,3,1).contiguous().view(features.shape[0],-1,4)
        return delta, target
