from unittest.mock import MagicMock
import pytest
import torch

from fasterrcnn.utils import RegionProposalNetwork


@pytest.fixture(scope='function')
def features():
    features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
    return features

@pytest.fixture(scope='function')
def anchor():
    anchor = MagicMock()
    anchor.anchors = torch.ones(10800, 4, dtype=torch.float32)
    anchor.train_nms_filter = [12000, 2000]
    anchor.test_nms_filter = [6000, 300]
    anchor.batch_size = [256, 64]
    anchor.roi_anchor_threshold = [0.5, 0.0]
    anchor.anchor_threshold=[0.5, 0.1]
    return anchor


class TestRPN(object):

    def test_rpn_0(self, features, anchor):
        rpn = RegionProposalNetwork(n_classes=2)
        delta, targets, loss = rpn(features, anchor, training=False)
        
        assert delta.shape == (1, 10800, 4)
        assert targets.shape == (1, 10800, 2)
        assert loss == None

    def test_rpn_1(self, features, anchor):
        true_targets = torch.tensor([[0, 0, 100, 100]])
        rpn = RegionProposalNetwork(n_classes=2)
        delta, targets, loss = rpn(features, anchor, true_targets, training=True)

        assert delta.shape == (1, 10800, 4)
        assert targets.shape == (1, 10800, 2)
        assert len(loss) == 2

    def test_generate_rpn_targets(self, anchor):
        true_targets = torch.tensor([[0, 0, 100, 100]])
        rpn = RegionProposalNetwork(n_classes=2)
        true_rpn_delta, true_rpn_targets, rpn_anchor_idx = rpn._generate_rpn_targets(anchor, true_targets)

        assert true_rpn_delta.shape == (256, 4)
        assert true_rpn_targets.shape == (256,)
        assert rpn_anchor_idx.shape == (256,)
        assert torch.count_nonzero(true_rpn_targets) == 0

    def test_compute_loss(self):
        true_rpn_delta = torch.tensor([[.1, .1, .1, .1]], dtype=torch.float)
        rpn_delta = torch.tensor([[[.5, .5, .5, .5]]], dtype=torch.float)
        true_rpn_targets = torch.tensor([1], dtype=torch.float)
        rpn_targets = torch.tensor([[1]], dtype=torch.float)
        rpn_idxs = [0]

        rcnn = RegionProposalNetwork(n_classes=2)
        loss = rcnn._compute_loss(
            true_rpn_delta, 
            true_rpn_targets, 
            rpn_idxs,
            rpn_delta, 
            rpn_targets
        )

        assert len(loss.keys()) == 2
        assert round(loss["rpn_bx_loss"].item(), 2) == 0.08
        assert loss["rpn_target_loss"].item() == 0