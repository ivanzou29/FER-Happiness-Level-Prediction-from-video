import sys
sys.path.insert(1, '/home/yunfan/FER-Happiness-Level-Prediction-from-video/video_image_manipulation')

import torch.nn as nn
import torch.nn.functional as F
import torch
import data_preprocessing
from torch.autograd import Variable

vgg_arch_16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
EPOCH = 20
BATCH_SIZE = 32
main_dir = "/data0/yunfan/frames"

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
        self.fc1 = nn.Linear(512*7*7, 5)

    def forward(self, x):
        x = x.float()
        for layer in self.vgg:
            x = layer(x)
        drop = self.conv_drop(x)
        del x

        drop_flatten = drop.view(-1, 512*7*7)

        del drop

        drop_flatten = F.relu(self.fc1(drop_flatten))

        return F.log_softmax(drop_flatten, dim=1)

class CompoundModel(nn.Module):
    def __init__(self):
        super(CompoundModel, self).__init__()
        self.cnn = CNN(vgg_arch_16)
        self.rnn = nn.LSTM(input_size= 5, hidden_size=10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(20, 1)

    def forward(self, x):
        batch_size = BATCH_SIZE
        time_step = 10
        C = 3
        H = 224
        W = 224
        c_in = x.view(batch_size * time_step, C, H, W)
        del x
        # print("c_in.shape: ")
        # print(c_in.shape)
        c_out = self.cnn(c_in)
        del c_in
        # print("c_out.shape: ")
        # print(c_out.shape)
        r_in = c_out.view(batch_size, time_step, -1)
        del c_out
        # print("r_in.shape: ")
        # print(r_in.shape)
        r_out, (h_n, h_c) = self.rnn(r_in, None)
        del r_in
        # print("r_out.shape: ")
        # print(r_out.shape)
        out = self.linear(r_out[:, -1, :])
        del r_out
        # print("out.shape: ")
        # print(out.shape)
        return out

if __name__ == '__main__':
    model = CompoundModel()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.L1Loss()
    my_data, my_label = data_preprocessing.get_all_data(main_dir)
    my_data = my_data[:, :, 0].astype(float)
    my_label = my_label.reshape(-1, 1)
    my_data, my_label = Variable(torch.from_numpy(my_data)).float().to(dtype=torch.float16), Variable(torch.from_numpy(my_label)).float().to(dtype=torch.float16)
    if torch.cuda.is_available():
        model = model.cuda()
        my_data = my_data.cuda()
        my_label = my_label.cuda()
    # print("my_data.shape: ")
    # print(my_data.shape)

    for i in range(EPOCH):
        optimizer.zero_grad()
        print("Epoch = " + str(i))
        f = open("dry_run3.txt", 'a')
        f.write("epoch: " + str(i) + "\n")
        starting_index = (i % 1) * BATCH_SIZE
        prediction = model(my_data)
        loss = loss_func(prediction, my_label)
        loss.backward()
        f.write("loss: " + str(loss) + "\n")
        print("loss: " + str(loss) + "\n")
        optimizer.step()
        f.close()

