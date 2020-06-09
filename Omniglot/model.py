import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128, resnet_depth=18):
        super(Model, self).__init__()

        self.f = []
        if resnet_depth == 18:
            my_resnet = resnet18()
            resnet_output_dim = 512
        elif resnet_depth == 34:
            my_resnet = resnet34()
            resnet_output_dim = 512
        elif resnet_depth == 50:
            my_resnet = resnet50()
            resnet_output_dim = 2048

        for name, module in my_resnet.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(resnet_output_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


# for 105x105 size
'''
class Omniglot_Model(nn.Module):
    def __init__(self):
        super(Omniglot_Model, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding=0), # out: 96
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 48
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=0), # out: 42
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 21
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0), # out: 18
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 9
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding=0), # out: 6
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 3
            nn.Flatten(),
            nn.Linear(9*128, 1024),
        )
        # projection head
        self.g = nn.Identity()

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
'''


# for 28x28 size (using max_pool)
class Omniglot_Model(nn.Module):
    def __init__(self):
        super(Omniglot_Model, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 14
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 14
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 7
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 7
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 3
            nn.Flatten(),
            nn.Linear(9*128, 1024),
        )
        # projection head
        self.g = nn.Identity()

    def forward(self, x, norm=True):
        feature = self.f(x)
        out = self.g(feature)
        if norm:
            return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        else:
            return F.normalize(feature, dim=-1), out


# for 28x28 size (not using maxpool)
'''
class Omniglot_Model(nn.Module):
    def __init__(self):
        super(Omniglot_Model, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1), # out: 14
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), # out: 14
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1), # out: 7
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), # out: 7
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0), # out: 3
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=2, stride=2), # out: 3
            nn.Flatten(),
            nn.Linear(9*128, 1024),
        )
        # projection head
        self.g = nn.Identity()

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
'''


# +
# for 28x28 size
class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class Recon_Omniglot_Model(nn.Module):
    def __init__(self):
        super(Recon_Omniglot_Model, self).__init__()
        # reconstructer (approximately the inverse of the encoder)
        self.f = nn.Sequential(
            nn.Linear(1024, 9*128, bias=False),
            nn.BatchNorm1d(num_features=9*128),
            nn.ReLU(inplace=True), # (9*128 -> 3*3*128)
            Lambda(lambda x: x.view(-1, 128, 3, 3)),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0,
                              output_padding=0, bias=False), # out: 7
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False), # out: 14
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                              output_padding=1, bias=False), # out: 28
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1,
                              output_padding=0, bias=True), # out: 28
            #nn.Sigmoid(),
        )

    def forward(self, x):
        recon = self.f(x)
        return recon


# -

# for 56x56 size
'''
class Omniglot_Model(nn.Module):
    def __init__(self):
        super(Omniglot_Model, self).__init__()
        # encoder
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 56
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 28
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 28
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 14
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 14
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 7
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), # out: 7
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # out: 3
            nn.Flatten(),
            nn.Linear(9*128, 1024),
        )
        # projection head
        self.g = nn.Identity()

    def forward(self, x):
        feature = self.f(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
'''
