from collections import namedtuple

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F

from .utils import Anchor, FeatureExtractor, RCNN, RegionProposalNetwork


class FasterRCNN(nn.Module):
    """ Function implements Faster-RCNN model from 
        https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf
    
    This class assumes the input data is in YOLO format 
                    (x_center, y_center, width, height)
    Structure:

        Feature Extraction:
            This process uses a CNN model to extract features from
            the input data.

        Region Proposal Network:
            This network takes the features from the feature extraction
            layer and maps the features to the image space using anchors.
            These anchors determine regions in the image which can be classified
            as background/foreground

        Region-based CNN:
            This network takes the highest anchor region proposals from
            the RPN and classifies the proposal as a class or background,
            and it determines bounding box locations for the proposals.
    """


    Loss = namedtuple("Loss",
        [
            "rpn_bx_loss",
            "rpn_target_loss",
            "roi_bx_loss",
            "roi_target_loss"
        ]
    )
    def __init__(self,
        backbone=None,
        n_classes=None,
        *args,
        **kargs
    ):
        super().__init__()
        n_anchors = 9
        # TODO: Adjust for different backbone models
        # TODO: adjust for different anchor sizes and shapes
        self.anchor = Anchor(self, n_anchors)
        self._feature_extractor = FeatureExtractor(pretrained=True)
        self._rpn = RegionProposalNetwork(n_classes=n_classes)
        self._rcnn = RCNN(self, n_classes=n_classes)
        n_anchors = 9
        self.rpn_batch_size = 256

    def loss_func(
        self, 
        true_rpn_bxs, 
        true_rpn_targets, 
        rpn_idxs,
        rpn_bxs, 
        rpn_targets,
        roi_bxs, 
        roi_targets, 
        true_roi_bxs, 
        true_roi_targets
    ):
        """ Function computes the loss for Faster RCNN Model

        Args:
            rpn_idxs (tensor.Tensor): RPN idxs
            rpn_bxs (tensor.Tensor): RPN model output delta boxes
            rpn_targets (tensor.Tensor): RPN model output targets
            true_rpn_bxs (tensor.Tensor): True delta boxes for RPN
            true_rpn_targets (tensor.Tensor): True targets for RPN
            roi_bxs (tensor.Tensor): ROI output delta boxes
            roi_targets (tensor.Tensor): ROI output targets
            true_roi_bxs (tensor.Tensor): True ROI delta boxes
            true_roi_targets (tensor.Tensor): True ROI targets

        Returns:
            namedtuple: output loss tuple  with idx names:        
                        'rpn_bx_loss',
                        'rpn_target_loss',
                        'roi_bx_loss',
                        'roi_target_loss'

            torch.Tensor: sum of losses used for backpropogation
        """
        rpn_bx_loss = F.smooth_l1_loss(rpn_bxs[0][rpn_idxs], true_rpn_bxs)
        rpn_target_loss = F.cross_entropy(rpn_targets[0][rpn_idxs], true_rpn_targets, ignore_index=-1)
        roi_bx_loss = F.smooth_l1_loss(roi_bxs, true_roi_bxs)
        roi_target_loss = F.cross_entropy(roi_targets, true_roi_targets, ignore_index=-1)
        losses = [rpn_bx_loss, rpn_target_loss, roi_bx_loss, roi_target_loss]
        return FasterRCNN.Loss(*losses), sum(losses)

    def forward(self, image_list, true_bx=None, true_targets=None):
        """ Forward pass over model

        Args:
            image_list (torch.Tensor): input images (b, c, w, h)
            true_bx (torch.Tensor): input images' true bounding boxes
            true_targets (torch.Tensor): input images' true class targets
        
        Returns:
            if training
                namedtuple: output loss tuple  with idx names:        
                        'rpn_bx_loss',
                        'rpn_target_loss',
                        'roi_bx_loss',
                        'roi_target_loss'

                torch.Tensor: sum of losses used for backpropogation
            
            if testing
                roi_bxs (torch.Tensor): (n,4) region boxes 
                roi_targets (torch.Tensor): (n, 4) class prediction targets
                    for rois
        """
        features = self._feature_extractor(image_list)
        rpn_delta, rpn_targets = self._rpn(features)
        anchors = self.anchor.generate_anchor_mesh(
            image_list, 
            features,
        )

        batch_rois, true_roi_delta, true_roi_targets = self.anchor.generate_roi(
            anchors,
            rpn_delta,
            rpn_targets,
            true_bx,
            true_targets,
            image_list.shape,
        )
        roi_delta, roi_targets = self._rcnn(features, batch_rois)
        if self.training:
            true_rpn_delta, true_rpn_targets, rpn_idxs = self.anchor.generate_rpn_targets(
                anchors,
                true_bx,
            )
            return self.loss_func(
                true_rpn_delta, 
                true_rpn_targets, 
                rpn_idxs,
                rpn_delta, 
                rpn_targets,
                roi_delta, 
                roi_targets, 
                true_roi_delta, 
                true_roi_targets
            )
        else:
            # TODO: adjust output to be unparameterized boxes from single class
            return roi_delta, roi_targets
