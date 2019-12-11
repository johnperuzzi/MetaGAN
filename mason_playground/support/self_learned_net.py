from torch import nn
import torch
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SelfLearnedNet(nn.Module):
    def __init__(self, n_way, device):
        super(SelfLearnedNet, self).__init__()
        self.latent_dim = 100

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


        # generator code
        self.device = device
        self.rand_hw_out = 7
        self.rand_ch_out = 64
        self.gen_net = []

        self.random_proj = nn.Linear(self.latent_dim, self.rand_hw_out*self.rand_hw_out*self.rand_ch_out)


        self.gen_net = nn.Sequential(
            nn.ConvTranspose2d(self.rand_ch_out, 32, 4, stride=2, padding=1), # [ch_in, ch_out, kernel_sz, stride, padding]
            nn.BatchNorm2d(32, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
            ).to(device)



    def gen(self, num):
        latent_rep = torch.randn(num, self.latent_dim, requires_grad=True, device=self.device)
        x = self.random_proj(latent_rep)
        x = F.relu(x, inplace=True)
        x = x.view(num, self.rand_ch_out, self.rand_hw_out, self.rand_hw_out)

        x = self.gen_net(x)
        return x


    def forward(self, x, cost=True):
        shared_rep = self.shared_net(x)
        nway_logits = self.nway_net(shared_rep)
        if not cost:
            return nway_logits

        gen_x = self.gen(x.size(0))
        shared_rep_gen = self.shared_net(x)


        return nway_logits, shared_rep, shared_rep_gen
        