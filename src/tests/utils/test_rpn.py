import pytest
import torch

from fasterrcnn.utils import RegionProposalNetwork


@pytest.fixture(scope='function')
def features():
    features = torch.ones(1, 512, 30, 40, dtype=torch.float32)
    return features


class TestRPN(object):

    def test_rpn_0(self, features):
        rpn = RegionProposalNetwork(n_classes=2)
        delta, targets = rpn(features)

        assert delta.shape == (1, 10800, 4)
        assert targets.shape == (1, 10800, 2)
