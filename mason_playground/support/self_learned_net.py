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

        self.cost_net = nn.Sequential(
            Flatten(),
            nn.Linear(64*3*3, 128),
            nn.BatchNorm1d(128, momentum=1, affine=True),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(128, 64), 
            # nn.BatchNorm1d(64, momentum=1, affine=True),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=1, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1),
            nn.BatchNorm1d(1, momentum=1, affine=True)).to(device)
            # maybe add batch norm to end to keep around 1?

    def forward(self, x):
        shared_rep = self.shared_net(x)
        nway_logits = self.nway_net(shared_rep)
        cost = self.cost_net(shared_rep)
        return nway_logits, cost
        