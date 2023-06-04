import pytest
import torch

from fasterrcnn import FasterRCNN


@pytest.fixture(scope='function')
def input_image():
    image = torch.ones(1, 3, 480, 640, dtype=torch.float32)
    return image


class TestFasterRCNN(object):

    def test_fasterrcnn_0(self, input_image):
        input_image = input_image
        true_bx = torch.tensor([[100, 100, 50, 50]])
        true_label = torch.tensor([1])
        model = FasterRCNN(n_classes=2)
        model.training = True

        losses = model(input_image, true_bx, true_label)
        
        assert len(losses) == 4

    def test_fasterrcnn_1(self, input_image):
        model = FasterRCNN(n_classes=2)
        model.training = False
        detections = model(input_image)

        assert isinstance(detections, list)

    def test_fasterrcnn_2(self, input_image):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_image = input_image.to(device)
        true_bx = torch.tensor([[100, 100, 50, 50]]).to(device)
        true_label = torch.tensor([1]).to(device)
        model = FasterRCNN(n_classes=2)
        model.to(device)
        
        losses = model(input_image, true_bx, true_label)
        
        assert len(losses) == 4