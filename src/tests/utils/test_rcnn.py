from unittest.mock import MagicMock
import pytest
import torch

from fasterrcnn.utils import RCNN


@pytest.fixture(scope='function')
def features():
    features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
    return features

@pytest.fixture(scope='function')
def anchor():
    anchor = MagicMock()
    anchor.anchors = torch.randint(0, 480, (10800, 4))
    anchor.train_nms_filter = [12000, 2000]
    anchor.test_nms_filter = [6000, 300]
    anchor.batch_size = [256, 64]
    anchor.roi_anchor_threshold = [0.5, 0.0]
    anchor.anchor_threshold=[0.5, 0.1]
    return anchor


class TestRCNN(object):

    def test_rcnn_0(self, features, anchor):
        rpn_delta = torch.randint(0, 480, (1, 10800, 4))
        rpn_targets = torch.zeros(1, 10800, 2)
        true_bx = torch.tensor([[0, 0, 100, 100]])
        true_targets = torch.tensor([0])
        img_shape = (1, 3, 480, 640)

        rcnn = RCNN(n_classes=2)
        detections, loss = rcnn(features,
            anchor, 
            rpn_delta, 
            rpn_targets, 
            img_shape,
            true_bx, 
            true_targets,
            training=True
        )

        assert detections == None
        assert len(loss.keys()) == 2

    def test_rcnn_1(self, features, anchor):
        rpn_delta = torch.randint(0, 480, (1, 10800, 4))
        rpn_targets = torch.zeros(1, 10800, 2)
        true_bx = torch.tensor([[0, 0, 100, 100]])
        true_targets = torch.tensor([0])
        img_shape = (1, 3, 480, 640)

        rcnn = RCNN(n_classes=2)
        detections, loss = rcnn(features,
            anchor, 
            rpn_delta, 
            rpn_targets, 
            img_shape,
            true_bx, 
            true_targets,
            training=False
        )

        assert isinstance(detections, list)
        assert loss == None

    def test_generate_roi(self, anchor):
        rpn_bxs = torch.randint(0, 480, (1, 10800, 4))
        rpn_targets = torch.zeros(1, 10800, 2)
        true_bx = torch.tensor([[0, 0, 100, 100]])
        true_targets = torch.tensor([0])
        img_shape = (1, 3, 480, 640)
        rcnn = RCNN(n_classes=2)
        batch_roi, true_roi_delta, true_roi_targets = rcnn._generate_roi(
            anchor, 
            rpn_bxs, 
            rpn_targets, 
            true_bx, 
            true_targets, 
            img_shape
        )

        assert batch_roi.shape == (256, 4)
        assert true_roi_delta.shape == (256, 4)
        assert true_roi_targets.shape == (256,)
    
    def test_compute_loss(self):
        true_roi_delta = torch.tensor([[.1, .1, .1, .1]], dtype=torch.float)
        roi_delta = torch.tensor([[.5, .5, .5, .5]], dtype=torch.float)
        true_roi_targets = torch.tensor([1], dtype=torch.float)
        roi_targets = torch.tensor([1], dtype=torch.float)
        rcnn = RCNN(n_classes=2)
        loss = rcnn._compute_loss(
            true_roi_delta, 
            true_roi_targets, 
            roi_delta, 
            roi_targets
        )

        assert len(loss.keys()) == 2
        assert round(loss["roi_bx_loss"].item(), 2) == 0.08
        assert loss["roi_target_loss"].item() == 0