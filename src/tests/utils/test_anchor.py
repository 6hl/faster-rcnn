from unittest.mock import MagicMock

import pytest
import torch

from fasterrcnn.utils import Anchor


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

    def test_generate_anchor_mesh(self, anchor):
        image = torch.ones(1, 3, 480, 640, dtype=torch.float32)
        features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
        anchors = anchor.forward(image, features)

        assert anchors.shape == (10800, 4)
