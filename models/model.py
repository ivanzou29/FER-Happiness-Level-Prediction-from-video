import torch
import torch.nn as nn


vgg_arch_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5']

input_channel = 3
image_size = (224, 224)
time_step = 16

input_dim = 3
output_dim = 1

# With reference to https://hellozhaozheng.github.io/z_post/PyTorch-VGG/

class model(nn.Module):
    def __init__(self, net_arch):
        super(model, self).__init__()
        self.net_arch = net_arch
        cnn_layers = []
        in_channels = 3

        for arch in net_arch:
            if arch == 'M':
                cnn_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                cnn_layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            else:
                cnn_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                cnn_layers.append(nn.ReLU(inplace=True))
                in_channels=arch

        self.vgg_face = nn.ModuleList(cnn_layers)
        self.vgg_nose = nn.ModuleList(cnn_layers)
        self.vgg_mouth = nn.ModuleList(cnn_layers)
        self.vgg_left_eye = nn.ModuleList(cnn_layers)
        self.vgg_right_eye = nn.ModuleList(cnn_layers)

        # TODO: define LSTM layers
        self.lstm_layers = 0



    def forward(self, input_data):
        face = input_data[:][0]
        nose = input_data[:][1]
        mouth = input_data[:][2]
        left_eye = input_data[:][3]
        right_eye = input_data[:][4]

        face = self.vgg_face(face)
        nose = self.vgg_nose(nose)
        mouth = self.vgg_mouth(mouth)
        left_eye = self.vgg_left_eye(left_eye)
        right_eye = self.vgg_right_eye(right_eye)

        x = torch.cat((face, nose, mouth, left_eye, right_eye), dim=2)

        # TODO: to feed the concatenated data into LSTM
        x = 0
        return x

