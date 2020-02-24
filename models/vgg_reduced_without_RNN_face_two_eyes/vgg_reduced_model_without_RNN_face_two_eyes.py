import sys
sys.path.insert(1, '/home/yunfan/FER-Happiness-Level-Prediction-from-video/video_image_manipulation')

import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import data_preprocessing_face_two_eyes

from math import sqrt as sqrt
from math import ceil as ceil
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = "Training batch size and epoch")
parser.add_argument('-b', '--batchsize', type=int, default=40, help='batch size')
parser.add_argument('-e', '--epoch', type=int, default=30, help='epoch')
parser.add_argument('-o', '--fcoutput', type=int, default=10, help='fc layer output num')
parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout proportion')
parser.add_argument('-t', '--timestep', type=int, default=10, help='time step')
parser.add_argument('-l', '--learningrate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-s', '--loss', type=int, default=1, help='loss function')

args = parser.parse_args()

my_vgg_arch = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']

EPOCH = args.epoch
BATCH_SIZE = args.batchsize
FC_LAYER_OUTPUT_NUM = args.fcoutput
DROPOUT = args.dropout
TIME_STEP = args.timestep
LEARNING_RATE = args.learningrate

loss_functions = ['MAE', 'RMSE']
LOSS = loss_functions[args.loss]

PREFIX = 'training_history_BatchSize=' + str(BATCH_SIZE) + '_Epoch=' + str(EPOCH) + '_FC_Num=' + str(FC_LAYER_OUTPUT_NUM) + '_TimeStep=' + str(TIME_STEP) + '_DropOut=' + str(DROPOUT) + '_LearningRate=' + str(LEARNING_RATE) + '_Loss=' + LOSS

main_dir = "/data0/yunfan/frames"
val_dir = "/data0/yunfan/val_frames"

HISTORY_DIR = "vgg_reduced_model_without_RNN_face_two_eyes_history/"

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
        self.conv_drop = nn.Dropout2d(p=DROPOUT)
        self.fc1 = nn.Linear(256*7*7, FC_LAYER_OUTPUT_NUM)

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
        self.cnn_0 = CNN(my_vgg_arch)
        self.cnn_1 = CNN(my_vgg_arch)
        self.cnn_2 = CNN(my_vgg_arch)
        self.linear = nn.Linear(3 * TIME_STEP * FC_LAYER_OUTPUT_NUM, 1)

    def forward(self, x):
        C = 3
        H = 56
        W = 56
        
        face = x[:, :, 0]
        region1 = x[:, :, 1]
        region2 = x[:, :, 2]

        del x

        face_sequence = []
        region1_sequence = []
        region2_sequence = []

        for i in range(TIME_STEP):
            face_frame = face[:,i]
            region1_frame = region1[:,i]
            region2_frame = region2[:,i]
            face_sequence.append(face_frame)
            region1_sequence.append(region1_frame)
            region2_sequence.append(region2_frame)

        del face
        del region1
        del region2

        face_out = []
        region1_out = []
        region2_out = []

        for i in range(len(face_sequence)):
            face_sequence[i] = face_sequence[i].view(-1, C, H, W)
            region1_sequence[i] = region1_sequence[i].view(-1, C, H, W)
            region2_sequence[i] = region2_sequence[i].view(-1, C, H, W)

            face_out.append(self.cnn_0(face_sequence[i]))
            region1_out.append(self.cnn_1(region1_sequence[i]))
            region2_out.append(self.cnn_2(region2_sequence[i]))

        del face_sequence
        del region1_sequence
        del region2_sequence

        face_out = torch.cat(tuple(face_out), 1)
        region1_out = torch.cat(tuple(region1_out), 1)
        region2_out = torch.cat(tuple(region2_out), 1)

        fc_out = torch.cat((face_out, region1_out, region2_out), 1)
        
        del face_out
        del region1_out
        del region2_out

        out = self.linear(fc_out)

        del fc_out

        return out

if __name__ == '__main__':
    model = CompoundModel()
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if (LOSS == 'RMSE'):
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.L1Loss()
    my_data, my_label = data_preprocessing_face_two_eyes.get_all_data(main_dir, TIME_STEP)
    val_data, val_label = data_preprocessing_face_two_eyes.get_all_data(val_dir, TIME_STEP)

    my_label = my_label.reshape(-1, 1)

    print(my_data.shape)
    print(my_label.shape)
    print(val_data.shape)
    print(val_label.shape)

    my_data, my_label = Variable(torch.from_numpy(my_data)).int(), Variable(torch.from_numpy(my_label)).float()

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

    os.system('mkdir ' + HISTORY_DIR + PREFIX + '/')

    f = open(HISTORY_DIR + PREFIX + '/history.txt', 'w+')
    for i in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            print('Epoch: ', i+1, '| Step: ', step)
            f.write('Epoch: '+ str(i+1) + '| Step: ' + str(step) + '\n')
            optimizer.zero_grad()

            training_prediction = model(batch_x)
            training_loss = criterion(training_prediction, batch_y)
            
            val_losses = []
            sizes = []
            for batch, (val_x, val_y) in enumerate(val_loader):
                val_prediction = model(val_x)
                val_loss = criterion(val_prediction, val_y).item()
                if (LOSS == 'RMSE'):
                    val_loss = sqrt(val_loss)
                val_losses.append(val_loss)
                sizes.append(len(val_y))

            val_loss_sum = 0
            val_loss_size = 0
            for i in range(len(val_losses)):
                val_loss_sum = val_loss_sum + val_losses[i] * sizes[i]
                val_loss_size = val_loss_size + sizes[i]

            val_loss_item = val_loss_sum / val_loss_size
            training_loss.backward()
            optimizer.step()

            if (LOSS == 'RMSE'):
                training_loss_item = sqrt(training_loss.item())
            else:
                training_loss_item = training_loss.item()

            training_loss_iteration.append(training_loss_item)
            validation_loss_iteration.append(val_loss_item)

            print('Training loss: ', str(training_loss_item))
            f.write('Training loss: ' +  str(training_loss_item) + '\n')
            print('Validation loss: ', str(val_loss_item))
            f.write('Validation loss: ' + str(val_loss_item) + '\n')

    iterations = len(training_loss_iteration)
    iteration = np.arange(iterations)

    f.close()

    training_loss_iteration = np.array(training_loss_iteration)
    validation_loss_iteration = np.array(validation_loss_iteration)
    
    plt.figure()

    train_history, = plt.plot(iteration, training_loss_iteration, 'r')
    val_history, = plt.plot(iteration, validation_loss_iteration, 'b')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend([train_history, val_history], ["training_loss", "validation_loss"], loc = 'upper right')
    history_plot_name = HISTORY_DIR + PREFIX + '/history_plot'
    plt.title('history plot')
    plt.savefig(history_plot_name + '.png')




    val_plot_data_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
    )
    val_prediction = []
    for batch, (val_x, val_y) in enumerate(val_plot_data_loader):
        val_prediction.append(model(val_x).cpu().detach())
    val_prediction = tuple(val_prediction)
    val_prediction = torch.cat(val_prediction, dim=0).numpy()
    val_label = (val_label.cpu()).numpy()
    val_prediction = val_prediction.reshape(len(val_prediction))
    val_label = val_label.reshape(len(val_label))
    
    plt.figure()
    val_standard_plot = plt.plot([0.0, 11.0], [0.0, 11.0], c='r', label='y=x')
    val_comparison_plot = plt.scatter(val_prediction, val_label, c='blue', alpha=0.6, label='Prediction vs Ground Truth')
    val_error_plot1 = plt.plot([0.0, 11.0], [1.0, 12.0], c='green', label='y=x+1', linestyle="-")
    val_error_plot2 = plt.plot([0.0, 11.0], [-1.0, 10.0], c='green', label='y=x-1', linestyle="-")
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.title('Validation label comparison')
    plt.savefig(HISTORY_DIR + PREFIX + '/validation_label_comparison.png')

    train_plot_data_loader = DataLoader(
            dataset=my_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
    )
    train_prediction = []
    for batch, (train_x, train_y) in enumerate(train_plot_data_loader):
        train_prediction.append(model(train_x).cpu().detach())
    train_prediction = tuple(train_prediction)
    train_prediction = torch.cat(train_prediction, dim=0).numpy()
    train_label = (my_label.cpu()).numpy()
    train_label = train_label.reshape(len(train_label))
    train_prediction = train_prediction.reshape(len(train_prediction))
    
    plt.figure()

    train_comparison_plot = plt.scatter(train_prediction, train_label, c='blue', alpha=0.6, label='Prediction vs Ground Truth')
    train_standard_plot = plt.plot([0.0, 11.0], [0.0, 11.0], 'r', label='y=x')
    train_error_plot1 = plt.plot([0.0, 11.0], [1.0, 12.0], c='green', label='y=x+1', linestyle="-")
    train_error_plot2 = plt.plot([0.0, 11.0], [-1.0, 10.0], c='green', label='y=x-1', linestyle="-")
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.title('Training label comparison')
    plt.savefig(HISTORY_DIR + PREFIX + '/training_label_comparison.png')



