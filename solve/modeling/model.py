import math

import torch
from torch import nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        return F.relu(self.linear(x_reshaped))


class SimpleModelPrelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        return self.prelu(self.linear(x_reshaped))
    

class SimpleModelBatchNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)
        self.batchnorm = nn.BatchNorm1d(10)

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        return F.relu(self.batchnorm(self.linear(x_reshaped)))


class SimpleModelBatchNormPrelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3*32*32, 10)
        self.batchnorm = nn.BatchNorm1d(10)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x_reshaped = x.reshape(x.shape[0], -1)
        return self.prelu(self.batchnorm(self.linear(x_reshaped)))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class ResnetBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(features)

    def forward(self, x):
        y = F.relu(self.batchnorm1(self.conv1(x)))
        z = self.batchnorm2(self.conv2(y))
        return F.relu(z + x)

class ResnetResizeBlock(nn.Module):
    def __init__(self, in_features, out_features, stride):
        super().__init__()
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, bias=False, stride=stride)
        self.batchnorm1 = nn.BatchNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_features)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_features))

    def forward(self, x):
        y = F.relu(self.batchnorm1(self.conv1(x)))
        x_down = self.downsample(x)
        z = self.batchnorm2(self.conv2(y))
        return F.relu(z + x_down)


class Resnet8(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(16)
        self.resblock1 = ResnetBlock(16)
        self.resblock2 = ResnetResizeBlock(16, 32, stride=2)
        self.resblock3 = ResnetResizeBlock(32, 64, stride=2)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm(self.conv(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = F.avg_pool2d(x, kernel_size=8)
        return self.linear(x.reshape(x.shape[0], -1))


class Resnet14(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(16)
        self.resblock1 = ResnetBlock(16)
        self.resblock2 = ResnetBlock(16)
        self.resblock3 = ResnetResizeBlock(16, 32, stride=2)
        self.resblock4 = ResnetBlock(32)
        self.resblock5 = ResnetResizeBlock(32, 64, stride=2)
        self.resblock6 = ResnetBlock(64)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm(self.conv(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = F.avg_pool2d(x, kernel_size=8)
        return self.linear(x.reshape(x.shape[0], -1))


class Resnet20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(16)
        self.resblock1 = ResnetBlock(16)
        self.resblock2 = ResnetBlock(16)
        self.resblock3 = ResnetBlock(16)
        self.resblock4 = ResnetResizeBlock(16, 32, stride=2)
        self.resblock5 = ResnetBlock(32)
        self.resblock6 = ResnetBlock(32)
        self.resblock7 = ResnetResizeBlock(32, 64, stride=2)
        self.resblock8 = ResnetBlock(64)
        self.resblock9 = ResnetBlock(64)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm(self.conv(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = F.avg_pool2d(x, kernel_size=8)
        return self.linear(x.reshape(x.shape[0], -1))
