import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


class Conditioner(nn.Module):
    """
    A model that takes in images and returns feature vectors for
    each one by passing them through some pretrained image classification
    model and extracting one of the last layers.
    """
    def __init__(self):
        super(Conditioner, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.eval()
        self.embedding = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        """
        Expects x to be of shape [batch_size, img_dim_1, img_dim_2] and of
        type torch.Tensor. Also assumes that x has already been normalized but
        not scaled.
        """
        resnet_input = nn.functional.interpolate(x, size=224)
        # Hacky way I suppose to deal with the possibility of 1 input channel or 3
        if resnet_input.shape[1] == 1:
            resnet_input = resnet_input.repeat(1, 3, 1, 1)
        return self.embedding(resnet_input)