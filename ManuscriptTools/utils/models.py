import torch
from torch import nn

resnet_50 = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)


class ResNet50Conv4(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Conv4, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])

    def forward(self, x):
        x = self.features(x)
        return x


def get_resnet50_conv4_model():
    conv4 = ResNet50Conv4(resnet_50)
    return conv4
