import pytest
import torch

from fasterrcnn.utils import RCNN


@pytest.fixture(scope='function')
def features():
    features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
    return features

@pytest.fixture(scope='function')
def roi():
    torch.manual_seed(0)
    roi = torch.randint(0, 480, (1000,4))
    return roi


class TestRCNN(object):

    def test_rcnn_0(self, features, roi):
        rcnn = RCNN(model=None, n_classes=2)
        delta, targets = rcnn(features, roi)

        assert delta.shape == (1000, 4)
        assert targets.shape == (1000, 3)
