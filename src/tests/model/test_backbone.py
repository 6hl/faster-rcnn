import pytest
import torch

from fasterrcnn.utils import FeatureExtractor


@pytest.fixture(scope='function')
def input_image():
    image = torch.ones(1, 3, 480, 640, dtype=torch.float32)
    return image


class TestBackbone(object):

    def test_backbone_0(self, input_image):
        extractor = FeatureExtractor()
        out = extractor(input_image)

        assert len(extractor._layers) == 30
        assert out.shape == (1, 512, 30, 40)
