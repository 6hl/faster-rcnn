import torch.nn as nn
import torchvision


class FeatureExtractor(nn.Module):
    """ Class used for feature extraction

    Args:
        pretrained (bool): used pretained model if True
                        else don't    
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # TODO: Allow for different backbones
        if pretrained:
            self._layers = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1).features[:30]
            for param in self._layers.parameters():
                param.requires_grad = False
        else:
            self._feat = torchvision.models.vgg16(weights=None).features[:30]
    
    def forward(self, x):
        """ Class is feature extractor backbone using vgg16
        Args:
            x (torch.Tensor): input image
        
        Returns:
            torch.Tensor: extracted features from input images
        """
        x = self._layers(x)
        return x