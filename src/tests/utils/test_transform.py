import pytest
import torch

from fasterrcnn.utils.transform import unparameterize, parameterize

class TestTransforms(object):
    
    def test_parameterize(self):
        source = torch.tensor([[0, 0, 10, 10]])
        dest = torch.tensor([[2, 2, 5, 5]])
        param = parameterize(source, dest)

        assert param.shape == (1, 4)
        assert param[0, 0].item() == 0.5
        assert param[0, 1].item() == 0.5
        assert round(param[0, 2].item(), 3) == 1.204
        assert round(param[0, 3].item(), 3) == 1.204

    def test_unparameterize(self):
        source = torch.tensor([[2, 2, 5, 5]])
        delta = torch.tensor([[0.5, 0.5, 1.204, 1.204]])
        unparam = unparameterize(source, delta)

        assert unparam.shape == (1,4)
        assert round(unparam[0, 0].item()) == 0
        assert round(unparam[0, 1].item()) == 0
        assert round(unparam[0, 2].item()) == 10
        assert round(unparam[0, 3].item()) == 10