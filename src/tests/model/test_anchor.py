from unittest.mock import MagicMock

import pytest
import torch

from fasterrcnn.utils import Anchor

torch.manual_seed(0)


@pytest.fixture(scope='function')
def anchor():
    model = MagicMock()
    model.training = True
    anchor = Anchor(model)
    return anchor

@pytest.fixture(scope='function')
def ratio_anchors():
    return torch.tensor(
        [
            [-1.5000,  0.5000,  5.5000,  3.5000],
            [-0.5000, -0.5000,  4.5000,  4.5000],
            [ 0.5000, -1.5000,  3.5000,  5.5000]
        ]
    )


class TestAnchor(object):

    def test_ratio_anchors(self, anchor):
        base_anchor = torch.tensor([0, 0, 5, 5])
        ratio_anchors = anchor._ratio_anchors(base_anchor, anchor.anchor_ratios)

        assert ratio_anchors.shape == (3, 4)

    def test_anchor_set(self, anchor):
        yolo_anchor = [
            torch.tensor([1]),
            torch.tensor([5]),
            torch.tensor([32, 64, 128]),
            torch.tensor([64, 128, 256])
        ]
        anchor_set = anchor._anchor_set(yolo_anchor)

        assert anchor_set.shape == (3, 4)

    def test_voc_to_yolo(self, anchor):
        bbx = torch.tensor([[30, 30, 100, 100]])
        yolo_anchor = anchor._voc_to_yolo(bbx)

        assert yolo_anchor.shape == (1,4)
        assert yolo_anchor[0, 0].item() == 79.5
        assert yolo_anchor[0, 1].item() == 79.5
        assert yolo_anchor[0, 2].item() == 71.0
        assert yolo_anchor[0, 3].item() == 71.0
        
    def test_scale_ratio_anchors(self, anchor, ratio_anchors):
        scaled_anchor = anchor._scale_ratio_anchors(ratio_anchors[0], anchor.scales)

        assert scaled_anchor.shape == (3,4)

    def test_parameterize(self, anchor):
        source = torch.tensor([[0, 0, 10, 10]])
        dest = torch.tensor([[2, 2, 5, 5]])
        param = anchor._parameterize(source, dest)

        assert param.shape == (1, 4)
        assert param[0, 0].item() == 0.5
        assert param[0, 1].item() == 0.5
        assert round(param[0, 2].item(), 3) == 1.204
        assert round(param[0, 3].item(), 3) == 1.204

    def test_unparameterize(self, anchor):
        source = torch.tensor([[2, 2, 5, 5]])
        delta = torch.tensor([[0.5, 0.5, 1.204, 1.204]])
        unparam = anchor._unparameterize(source, delta)

        assert unparam.shape == (1,4)
        assert round(unparam[0, 0].item()) == 0
        assert round(unparam[0, 1].item()) == 0
        assert round(unparam[0, 2].item()) == 10
        assert round(unparam[0, 3].item()) == 10

    def test_generate_anchor_mesh(self, anchor):
        image = torch.ones(1, 3, 480, 640, dtype=torch.float32)
        features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
        anchors = anchor.generate_anchor_mesh(image, features)

        assert anchors.shape == (10800, 4)
    
    def test_generate_rpn_targets(self, anchor):
        anchors = torch.ones(10800, 4, dtype=torch.float32)
        true_targets = torch.tensor([[0, 0, 100, 100]])
        true_rpn_delta, true_rpn_targets, rpn_anchor_idx = anchor.generate_rpn_targets(anchors, true_targets)

        assert true_rpn_delta.shape == (256, 4)
        assert true_rpn_targets.shape == (256,)
        assert rpn_anchor_idx.shape == (256,)
        assert torch.count_nonzero(true_rpn_targets) == 0

    def test_generate_roi(self, anchor):
        anchors = torch.randint(0, 480, (10800, 4))
        rpn_bxs = torch.randint(0, 480, (1, 10800, 4))
        rpn_targets = torch.zeros(1, 10800, 2)
        true_bx = torch.tensor([[0, 0, 100, 100]])
        true_targets = torch.tensor([0])
        img_shape = (1, 3, 480, 640)
        batch_roi, true_roi_delta, true_roi_targets = anchor.generate_roi(anchors, rpn_bxs, rpn_targets, true_bx, true_targets, img_shape)

        assert batch_roi.shape == (256, 4)
        assert true_roi_delta.shape == (256, 4)
        assert true_roi_targets.shape == (256,)
