import torch.nn as nn
import torch.nn.functional as F
import torch
from video_image_manipulation import data_preprocessing
from torch.autograd import Variable

vgg_arch_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
epoch = 200
main_dir = ""

class CNN(nn.Module):
    def __init__(self, net_arch):
        super(CNN, self).__init__()
        layers = []
        in_channels = 3
        for arch in net_arch:
            if arch == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif arch == 'M5':
                layers.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
            elif arch == "FC1":
                layers.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC2":
                layers.append(nn.Conv2d(1024, 1024, kernel_size=1))
                layers.append(nn.ReLU(inplace=True))
            elif arch == "FC":
                layers.append(nn.Conv2d(1024, 10, kernel_size=1))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch

        self.vgg = nn.ModuleList(layers)
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(50176, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        for layer in self.vgg:
            x = layer(x)
        x = self.conv_drop(x)
        x = x.view(-1, 50176)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

class CompoundModel(nn.Module):
    def __init__(self):
        super(CompoundModel, self).__init__()
        self.cnn = CNN(vgg_arch_16)
        self.rnn = nn.LSTM(input_size=10, hidden_size=5, num_layers=1, batch_first=True)
        self.linear = nn.Linear(5, 1)

    def forward(self, x):
        batch_size = 1
        time_step = 16
        C = 3
        H = 224
        W = 224
        print(x.shape)
        c_in = x.view(batch_size * time_step, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, time_step, -1)
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear(r_out[:, -1, :])

        return r_out2

if __name__ == '__main__':
    model = CompoundModel()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    loss_func = torch.nn.MSELoss()
    my_data, my_label = data_preprocessing.get_all_data(main_dir)
    my_data = my_data[:, :, 0]
    my_data, my_label = Variable(torch.from_numpy(my_data)), Variable(torch.from_numpy(my_label))

    for i in range(epoch):
        prediction = model(my_data[epoch % 40])
        loss = loss_func(prediction, my_label[epoch % 40])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
