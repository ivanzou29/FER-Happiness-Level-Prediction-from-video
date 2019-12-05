import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


vgg_arch_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5']

class CNN_Net(nn.Module):
    def __init__(self, ):
        super(CNN_Net, self).__init__()
        layers = []
        in_channels = 3

        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels=arch
        self.vgg = nn.ModuleList(layers)

    def forward(self, input_data):
        x_face = input_data[0]
        x_nose = input_data[1]
        x_mouth = input_data[2]
        x_left_eye = input_data[3]
        x_right_eye = input_data[4]

        for layer in self.vgg:
            x_face = layer(x_face)
            x_nose = layer(x_nose)
            x_mouth = layer(x_mouth)
            x_left_eye = layer(x_left_eye)
            x_right_eye = layer(x_right_eye)

        x = torch.cat((x_face, x_nose, x_mouth, x_left_eye, x_right_eye), dim=2)

        return x

