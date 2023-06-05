from typing import Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from .anchor import Anchor
from .transform import parameterize, unparameterize


class RCNN(nn.Module):
    """ Region based CNN

    Args:
        n_classes (int): number of classes
        pool_size (int): feature map pooling size
        pretrained (bool): use pretrained feature maps
    """
    def __init__(
        self, 
        n_classes: int, 
        pool_size: int = 7, 
        pretrained: bool = True
    ):
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
    
    def forward(
        self, 
        features: torch.Tensor, 
        anchor: Anchor, 
        rpn_delta: torch.Tensor, 
        rpn_targets: torch.Tensor, 
        img_shape: torch.Tensor,
        true_bx: Union[torch.Tensor, None] = None, 
        true_targets: Union[torch.Tensor, None] = None, 
        training: bool = True
    ):
        """ Forward pass of Region Based CNN

        Args:
            features (torch.Tensor): extracted features (B, C, H, W)
            anchor (nn.Module): anchor object
            rpn_delta (torch.Tensor): region proposal network delta
            rpn_targets (torch.Tensor): region proposal network targets
            img_shape (torch.Tensor): input image shape
            true_bx (Union[torch.Tensor, None]): true bounding boxes for input image
            true_targets (Union[torch.Tensor, None]): true targets for input image
            training (bool): if model is training
        
        Returns:
            if training:
                tuple (None, dict[str, torch.Tensor]): dict contains losses
            else:
                tuple (dict[str, torch.Tensor]): dict contains bbx and target
        """
        self.training = training
        roi, true_roi_delta, true_roi_targets = self._generate_roi(
            anchor, 
            rpn_delta, 
            rpn_targets, 
            true_bx, 
            true_targets, 
            img_shape
        )

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

        if not self.training:
            idxs = targets.max(dim=1)[1] > 0
            return [{"bbx": bx, "target": target} for bx, target in zip(unparameterize(roi, delta)[idxs], targets[idxs])], None

        loss = self._compute_loss(true_roi_delta, true_roi_targets, delta, targets)

        return None, loss
    
    def _post_process(self, onehot_delta, targets):
        _, max_idx = targets.max(dim=1)
        delta = torch.stack(
            [bx[idx*4:idx*4+4] for bx, idx in zip(onehot_delta, max_idx)],
            dim=0
        )
        return delta
    
    def _compute_loss(
        self, 
        true_roi_delta, 
        true_roi_targets, 
        roi_delta, 
        roi_targets
    ):
        roi_bx_loss = F.smooth_l1_loss(roi_delta, true_roi_delta)
        roi_target_loss = F.cross_entropy(roi_targets, true_roi_targets, ignore_index=-1)
        return {"roi_bx_loss": roi_bx_loss, "roi_target_loss": roi_target_loss}
    
    def _generate_roi(
        self, 
        anchor: Anchor, 
        rpn_delta: torch.Tensor, 
        rpn_targets: torch.Tensor, 
        true_bx: torch.Tensor, 
        true_targets: torch.Tensor, 
        img_shape: torch.Tensor
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        
        """ Generates ROI
        Args:
            anchor (nn.Module): anchor object
            rpn_delta (torch.Tensor[1, N, 4]): RPN anchor deltas
            rpn_targets (torch.Tensor[1, N, C]): RPN targets
            true_bx (torch.Tensor[N, 4]): ground truth bbxs
            true_targets (torch.Tensor[N, C]): ground truth targets
            img_shape (torch.Tensor[1,3,H,W]): input image shape
        Returns:
            tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]
        """

        anchors = anchor.anchors
        if self.training:
            nms_filter = anchor.train_nms_filter
        else:
            nms_filter = anchor.test_nms_filter
        
        rpn_un_param = unparameterize(anchors, rpn_delta[0].clone().detach())
        rpn_anchors = torchvision.ops.clip_boxes_to_image(rpn_un_param, (img_shape[2], img_shape[3]))
        rpn_anchor_idx = torchvision.ops.remove_small_boxes(rpn_anchors, 16.0)
        rpn_anchors = rpn_anchors[rpn_anchor_idx, :]
        rpn_targets = rpn_targets[0][rpn_anchor_idx].clone().detach().view(-1, 2)[:,1]

        top_scores_idx = rpn_targets.argsort()[:nms_filter[0]]
        rpn_anchors = rpn_anchors[top_scores_idx, :].to(torch.float64)
        rpn_targets = rpn_targets[top_scores_idx].to(torch.float64)

        nms_idx = torchvision.ops.nms(rpn_anchors, rpn_targets, iou_threshold=0.7)
        nms_idx = nms_idx[:nms_filter[1]]
        rpn_anchors = rpn_anchors[nms_idx, :]
        rpn_targets = rpn_targets[nms_idx]

        if not self.training:
            return rpn_anchors, None, None

        anchor_iou = torchvision.ops.box_iou(true_bx.reshape(-1,4), rpn_anchors)
        max_values, max_idx = anchor_iou.max(dim=0)
        true_roi_targets = torch.zeros(anchor.batch_size)
        
        # Find fg/bg anchor idx
        fg_idx = (max_values >= anchor.anchor_threshold[0]).nonzero().ravel()
        bg_idx = ((max_values < anchor.roi_anchor_threshold[0]) &
                (max_values >= anchor.roi_anchor_threshold[1])).nonzero().ravel()

        # Create batch from fg/bg idx, consider repeated values
        if len(fg_idx) > 0:
            fg_idx = fg_idx[
                torch.ones(len(fg_idx)).multinomial(
                    min(int(anchor.batch_size[0]/2), len(fg_idx)), 
                    replacement=False
                )
            ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                anchor.batch_size[0]-len(fg_idx), 
                replacement=True if len(bg_idx) <= anchor.batch_size[0]-len(fg_idx) else False
            )
        ]
        
        batch_rpn_idx = torch.cat((fg_idx, bg_idx), dim=0)
        batch_roi = rpn_anchors[batch_rpn_idx]
        true_roi_delta = parameterize(batch_roi, anchors[batch_rpn_idx])
        true_roi_targets = torch.cat((true_targets[max_idx[fg_idx]], torch.zeros(len(bg_idx), dtype=torch.long).to(true_roi_delta.device))).long()
        return batch_roi, true_roi_delta, true_roi_targets