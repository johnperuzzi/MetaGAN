from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SelfLearnedNet(nn.Module):
    def __init__(self, n_way, device):
        super(SelfLearnedNet, self).__init__()

        self.shared_net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True)).to(device)


        self.nway_net = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            Flatten(),
            nn.Linear(64, n_way)).to(device)

    def forward(self, x, cost=True):
        shared_rep = self.shared_net(x)
        nway_logits = self.nway_net(shared_rep)

        return nway_logits, shared_rep if cost else nway_logits
        