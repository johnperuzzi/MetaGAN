import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


class SimpleGenerator(nn.Module):
    """
    A model that takes in images and returns feature vectors for
    each one by passing them through some pretrained image classification
    model and extracting one of the last layers.
    """
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        """
        Expects x to be of shape [batch_size, img_dim_1, img_dim_2] and of
        type torch.Tensor. Also assumes that x has already been normalized but
        not scaled.
        """
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        return x, y