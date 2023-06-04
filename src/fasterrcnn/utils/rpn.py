from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from .transform import parameterize


class RegionProposalNetwork(nn.Module):
    """ Class used to map extracted features to image space using anchors """
    def __init__(
        self, 
        n_classes, 
        channels=512, 
        n_anchors=9
    ):
        super().__init__()
        self.n_classes = n_classes
        self._conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=1, padding="same")
        self._target = nn.Conv2d(in_channels=channels, out_channels=n_anchors*2, kernel_size=(1,1), stride=1, padding=0)
        self._delta = nn.Conv2d(in_channels=channels, out_channels=n_anchors*4, kernel_size=(1,1), stride=1, padding=0)

    def forward(self, features, anchors, true_bx=None, training=True):
        """ Forward pass of RPN

        Args:
            features (torch.Tensor): extracted features
        
        Returns:
            torch.Tensor: bounding deltas (b, N, 4)
            torch.Tensor: predicted targets fg/bg (b, N, 2)
        """
        self.training = training
        x = F.relu(self._conv(features))
        target = self._target(x).permute(0, 2, 3, 1).contiguous()
        target = F.softmax(target.view(features.shape[0], -1, 2), dim=2)
        delta = self._delta(x).permute(0,2,3,1).contiguous().view(features.shape[0],-1,4)

        if not self.training:
            return delta, target, None
        
        true_rpn_delta, true_rpn_targets, rpn_anchor_idx = self._generate_rpn_targets(anchors, true_bx)
        
        loss = self._compute_loss(true_rpn_delta, true_rpn_targets, rpn_anchor_idx, delta, target)
        return delta, target, loss
    
    def _compute_loss(
        self, 
        true_rpn_delta, 
        true_rpn_targets, 
        rpn_idxs,
        rpn_delta, 
        rpn_targets
    ):
        rpn_bx_loss = F.smooth_l1_loss(rpn_delta[0][rpn_idxs], true_rpn_delta)
        rpn_target_loss = F.cross_entropy(rpn_targets[0][rpn_idxs], true_rpn_targets, ignore_index=-1)
        return {"rpn_bx_loss": rpn_bx_loss, "rpn_target_loss": rpn_target_loss}
    
    def _generate_rpn_targets(
        self,
        anchor, 
        true_bx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        """ Generates RPN Targets

        Args:
            anchors (torch.tensor[N, 4]): generated anchors
            true_bx (torch.tensor[N, 4]): ground truth bbxs
        
        Returns:
            true_rpn_delta (torch.tensor[N, 4])
            true_rpn_targets (torch.tensor[N, 4])
            rpn_anchor_idx (torch.tensor[N, 4])
        """
        
        anchors = anchor.anchors
        true_bx = true_bx.reshape(-1,4)
        anchor_iou = torchvision.ops.box_iou(true_bx, anchors)
        max_values, max_idx = anchor_iou.max(dim=0)
        true_anchor_bxs = torch.stack(
            [true_bx[m.item()] for m in max_idx],
            dim=0
        )

        # Find fg/bg anchor idx
        fg_idx = (max_values >= anchor.anchor_threshold[0]).nonzero().ravel()
        bg_idx = (max_values <= anchor.anchor_threshold[1]).nonzero().ravel()
        bg_bool = True if len(bg_idx) <= int(anchor.batch_size[0]/2) else False

        # Create batch from fg/bg idx
        if len(fg_idx) > 0:
            fg_idx = fg_idx[
                torch.ones(len(fg_idx)).multinomial(
                    min(int(anchor.batch_size[0]/2), len(fg_idx)),
                    replacement=False)
                ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                anchor.batch_size[0]-len(fg_idx),
                replacement=bg_bool)
            ]
        
        rpn_anchor_idx = torch.cat((fg_idx, bg_idx), dim=0)
        true_rpn_delta = parameterize(anchors[rpn_anchor_idx, :], true_anchor_bxs[rpn_anchor_idx, :])
        true_rpn_targets = torch.tensor([1]*len(fg_idx) + [0]*len(bg_idx)).to(true_rpn_delta.device)
        return true_rpn_delta, true_rpn_targets, rpn_anchor_idx
