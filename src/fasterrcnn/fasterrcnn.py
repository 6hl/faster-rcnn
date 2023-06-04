from collections import namedtuple

import torch
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

    def __init__(self,
        backbone=None,
        n_classes=None,
        n_anchors = 9,
        *args,
        **kargs
    ):
        super().__init__()
        self._anchor = Anchor(n_anchors=n_anchors)
        self._feature_extractor = FeatureExtractor(backbone, pretrained=True)
        self._rpn = RegionProposalNetwork(n_classes=n_classes)
        self._rcnn = RCNN(n_classes=n_classes)

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
        
        # TODO: Consider dynamic anchor generation
        self._anchor.forward(image_list, features)
        
        rpn_delta, rpn_targets, rpn_loss = self._rpn(features, self._anchor, true_bx, self.training)

        detections, roi_loss = self._rcnn(
            features, 
            self._anchor, 
            rpn_delta, 
            rpn_targets,
            image_list.shape, 
            true_bx, 
            true_targets,
            self.training
        )

        if self.training:
            return {**rpn_loss, **roi_loss}
        return detections
