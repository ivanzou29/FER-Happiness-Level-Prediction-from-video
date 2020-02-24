import sys
sys.path.insert(1, '/home/yunfan/FER-Happiness-Level-Prediction-from-video/video_image_manipulation')

import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import data_preprocessing_light

from math import sqrt as sqrt
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = "Training batch size and epoch")
parser.add_argument('-b', '--batchsize', type=int, default=40, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=30, help='epoch')
args = parser.parse_args()

my_vgg_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
EPOCH = args.epoch
BATCH_SIZE = args.batchsize
main_dir = "/data0/yunfan/frames"
val_dir = "/data0/yunfan/val_frames"

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
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=arch, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels = arch

        self.vgg = nn.ModuleList(layers)
        self.conv_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(256*7*7, 10)

    def forward(self, x):
        x = x.float()
        for layer in self.vgg:
            x = layer(x)
        drop = self.conv_drop(x)
        del x

        drop_flatten = drop.view(-1, 256*7*7)

        del drop

        drop_flatten = F.relu(self.fc1(drop_flatten), inplace=True)

        return F.log_softmax(drop_flatten, dim=1)

class CompoundModel(nn.Module):
    def __init__(self):
        super(CompoundModel, self).__init__()
        self.cnn = CNN(my_vgg_arch)
        self.rnn = nn.LSTM(input_size= 10, hidden_size=10, num_layers=1, batch_first=True)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        batch_size = BATCH_SIZE
        time_step = 10
        C = 3
        H = 56
        W = 56
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    my_data, my_label = data_preprocessing_light.get_all_data(main_dir)
    val_data, val_label = data_preprocessing_light.get_all_data(val_dir)
    my_data = my_data[:, :, 0]
    my_label = my_label.reshape(-1, 1)

    print(np.any(np.isnan(my_data)))
    print(np.any(np.isnan(my_label)))
    my_data, my_label = Variable(torch.from_numpy(my_data)).int(), Variable(torch.from_numpy(my_label)).float()
    val_data = val_data[:, :, 0]
    val_label = val_label.reshape(-1, 1)
    val_data, val_label = Variable(torch.from_numpy(val_data)).int(), Variable(torch.from_numpy(val_label)).float()
    if torch.cuda.is_available():
        model = model.cuda()
        my_data = my_data.cuda()
        my_label = my_label.cuda()
        val_data = val_data.cuda()
        val_label = val_label.cuda()

    my_dataset = TensorDataset(my_data, my_label)
    train_loader = DataLoader(
            dataset=my_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
    )

    val_dataset = TensorDataset(val_data, val_label)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
    )

    training_loss_iteration = []
    validation_loss_iteration = []

    f = open('training_history_BatchSize=' + str(BATCH_SIZE) + '_Epoch=' + str(EPOCH) + '.txt', 'w+')

    for i in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            print('Epoch: ', i+1, '| Step: ', step)
            f.write('Epoch: '+ str(i+1) + '| Step: ' + str(step) + '\n')
            optimizer.zero_grad()
            training_prediction = model(batch_x)
            training_loss = criterion(training_prediction, batch_y)
            
            val_losses = []
            for batch, (val_x, val_y) in enumerate(val_loader):
                if (batch == len(val_loader) - 1):
                    break
                val_prediction = model(val_x)
                val_loss = criterion(val_prediction, val_y)
                val_losses.append(sqrt(val_loss.item()))
            
            val_losses = np.array(val_losses)
            val_loss = np.mean(val_losses)
            training_loss.backward()
            optimizer.step()
            training_loss_iteration.append(sqrt(training_loss.item()))
            validation_loss_iteration.append(val_loss)
            print('Training loss: ', sqrt(training_loss.item()))
            f.write('Training loss: ' + str(sqrt(training_loss.item())) + '\n')
            print('Validation loss: ', val_loss)
            f.write('Validation loss: ' + str(val_loss) + '\n')

    iterations = len(training_loss_iteration)
    iteration = np.arange(iterations)

    f.close()

    training_loss_iteration = np.array(training_loss_iteration)
    validation_loss_iteration = np.array(validation_loss_iteration)

    train_history, = plt.plot(iteration, training_loss_iteration, 'r')
    val_history, = plt.plot(iteration, validation_loss_iteration, 'b')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend([train_history, val_history], ["training_loss", "validation_loss"], loc = 'upper right')
    name = 'training_history_BatchSize=' + str(BATCH_SIZE) + '_Epoch=' + str(EPOCH) 
    plt.title(name)
    plt.savefig(name + '.png')
    


