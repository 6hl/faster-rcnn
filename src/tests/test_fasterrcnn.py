import pytest
import torch

from fasterrcnn import FasterRCNN


@pytest.fixture(scope='function')
def input_image():
    image = torch.ones(1, 3, 480, 640, dtype=torch.float32)
    return image


class TestFasterRCNN(object):

    def test_fasterrcnn_0(self, input_image):
        true_bx = torch.tensor([[100, 100, 50, 50]])
        true_label = torch.tensor([1])
        model = FasterRCNN(n_classes=2)
        losses, sum_losses = model(input_image, true_bx, true_label)
        
        assert len(losses) == 4

    def test_fasterrcnn_1(self, input_image):
        model = FasterRCNN(n_classes=2)
        model.training = False
        roi_delta, roi_targets = model(input_image)

        assert roi_delta.shape == (300, 4)
        assert roi_targets.shape == (300, 3)