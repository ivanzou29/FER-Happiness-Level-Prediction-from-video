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
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = "training batch size and epoch")
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

LOSS_FUNCTIONS = ['MAE', 'RMSE']
LOSS = LOSS_FUNCTIONS[args.loss]

# select the loss function
if (LOSS == 'RMSE'):
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.L1Loss()

PREFIX = 'training_history_BatchSize=' + str(BATCH_SIZE) + '_Epoch=' + str(EPOCH) + '_FC_Num=' + str(FC_LAYER_OUTPUT_NUM) + '_TimeStep=' + str(TIME_STEP) + '_DropOut=' + str(DROPOUT) + '_LearningRate=' + str(LEARNING_RATE) + '_Loss=' + LOSS

TRAIN_VAL_DIR = "/data0/yunfan/train_frames"
TEST_DIR = "/data0/yunfan/test_frames"

HISTORY_DIR = "vgg_reduced_model_without_RNN_history/"

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
        self.cnn = CNN(my_vgg_arch)
        self.linear = nn.Linear(TIME_STEP * FC_LAYER_OUTPUT_NUM, 1)

    def forward(self, x):
        C = 3
        H = 56
        W = 56

        face = x

        del x

        image_sequence = []
        
        for i in range(TIME_STEP):
            face_frame = face[:,i]
            image_sequence.append(face_frame)

        del face

        image_out = []
        for face_frame in image_sequence:
            
            face_frame = face_frame.view(-1, C, H, W)

            face_frame_out = self.cnn(face_frame)
            image_out.append(face_frame_out)

        del image_sequence

        image_out = tuple(image_out)

        fc_out = torch.cat(image_out, 1)
        
        del image_out

        out = self.linear(fc_out)

        del fc_out

        return out

def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def set_bn_train(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.train()

def train_val_split(my_data, my_label):
    train_data, val_data, train_label, val_label = train_test_split(my_data, my_label, test_size = 0.30, random_state = 42)
    train_label = train_label.reshape(-1, 1)
    val_label = val_label.reshape(-1, 1)
    return train_data, train_label, val_data, val_label

def training(model, train_loader, val_loader, test_loader):

    # prepare for training
    training_loss_iteration = []
    validation_loss_iteration = []
    os.system('mkdir ' + HISTORY_DIR + PREFIX + '/')
    f = open(HISTORY_DIR + PREFIX + '/history.txt', 'w+')

    # training
    for i in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            print('Epoch: ', i+1, '| Step: ', step)
            f.write('Epoch: '+ str(i+1) + '| Step: ' + str(step) + '\n')
            optimizer.zero_grad()

            training_prediction = model(batch_x)
            training_loss = criterion(training_prediction, batch_y)
            
            print(training_prediction)
            print(batch_y)            
            val_losses = []
            sizes = []
            for batch, (val_x, val_y) in enumerate(val_loader):
                val_prediction = model(val_x)
                val_loss = criterion(val_prediction, val_y).item()
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
                val_loss_item = sqrt(val_loss_item)
            else:
                training_loss_item = training_loss.item()

            training_loss_iteration.append(training_loss_item)
            validation_loss_iteration.append(val_loss_item)

            print('Training loss: ', str(training_loss_item))
            f.write('Training loss: ' +  str(training_loss_item) + '\n')
            print('Validation loss: ', str(val_loss_item))
            f.write('Validation loss: ' + str(val_loss_item) + '\n\n')

    test_losses = []
    test_sizes = []
    for batch, (test_x, test_y) in enumerate(test_loader):
        test_prediction = model(test_x)
        test_loss = criterion(test_prediction, test_y).item()
        test_losses.append(test_loss)
        test_sizes.append(len(test_y))

    test_loss_sum = 0
    test_loss_size = 0
    for i in range(len(test_losses)):
        test_loss_sum = test_loss_sum + test_losses[i] * test_sizes[i]
        test_loss_size = test_loss_size + test_sizes[i]

    test_loss_item = test_loss_sum / test_loss_size
    if (LOSS == 'RMSE'):
        test_loss_item = sqrt(test_loss_item)
    print('Testing loss: ' + str(test_loss_item))
    f.write('Testing loss: ' + str(test_loss_item) + '\n')
    f.close()


    iterations = len(training_loss_iteration)
    iteration = np.arange(iterations)
    training_loss_iteration = np.array(training_loss_iteration)
    validation_loss_iteration = np.array(validation_loss_iteration)
    
    return training_loss_iteration, validation_loss_iteration, iteration


def plot_training_history(training_loss_iteration, validation_loss_iteration, iteration):
    plt.style.use('ggplot')
    plt.figure()

    train_history, = plt.plot(iteration, training_loss_iteration, 'r')
    val_history, = plt.plot(iteration, validation_loss_iteration, 'b')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend([train_history, val_history], ["training_loss", "validation_loss"], loc = 'upper right')
    history_plot_name = HISTORY_DIR + PREFIX + '/history_plot'
    plt.title('history plot')
    plt.savefig(history_plot_name + '.png')

# plot the comparison between predicted result and ground truth 
def plot_comparison_pred_gt(model, dataset_name, label, plot_type):
    plot_data_loader = DataLoader(
            dataset=dataset_name,
            batch_size=BATCH_SIZE,
            shuffle=False
    )
    prediction = []
    losses = []
    sizes = []
    for batch, (batch_x, batch_y) in enumerate(plot_data_loader):
        pred = model(batch_x).cpu().detach()
        prediction.append(pred)
    
    prediction = tuple(prediction)
    prediction = torch.cat(prediction, dim=0).numpy()
    label = (label.cpu()).numpy()
    prediction = prediction.reshape(len(prediction))
    label = label.reshape(len(label))

    error_within_one_percent, error_within_two_percent = count_error(prediction, label)

    plt.style.use('ggplot')
    plt.figure()
    plt.plot([0.0, 11.0], [0.0, 11.0], 'g')
    plt.scatter(label, prediction, c='b', alpha=0.6)
    plt.plot([0.0, 11.0], [1.0, 12.0], 'y--')
    plt.plot([0.0, 11.0], [-1.0, 10.0], 'm--')
    plt.plot([0.0, 11.0], [2.0, 13.0], 'c--')
    plt.plot([0.0, 11.0], [-2.0, 9.0], 'r--')
    plt.legend(['y=x', 'y=x+1', 'y=x-1', 'y=x+2', 'y=x-2'], loc='upper left')
    plt.text(7,1.5, 'Error <= 1 rate: ' + str(error_within_one_percent)[:5] + '%', fontsize=10)
    plt.text(7,0.5, 'Error <= 2 rate: ' + str(error_within_two_percent)[:5] + '%', fontsize=10)


    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    plt.title(plot_type + ' result comparison')
    plt.savefig(HISTORY_DIR + PREFIX + '/' + plot_type + '_result_comparison.png')

def count_error(prediction, label):
    error_within_one = 0
    error_within_two = 0
    errors = prediction - label
    for error in errors:
        if (abs(error) <= 1):
            error_within_one = error_within_one + 1
            error_within_two = error_within_two + 1
        elif ((abs(error) > 1) and (abs(error) <= 2)):
            error_within_two = error_within_two + 1
    error_within_one_percent = (error_within_one * 100) / len(errors)
    error_within_two_percent = (error_within_two * 100) / len(errors)

    return error_within_one_percent, error_within_two_percent

if __name__ == '__main__':
    # basic model and optimizer setup
    model = CompoundModel()
    model.apply(set_bn_eval)
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # get the training and validation data
    my_data, my_label = data_preprocessing_face_two_eyes.get_all_data(TRAIN_VAL_DIR, TIME_STEP)
    my_data = my_data[:, :, 0]
    train_data, train_label, val_data, val_label = train_val_split(my_data, my_label)
    del my_data
    del my_label

    # get the testing data
    test_data, test_label = data_preprocessing_face_two_eyes.get_all_data(TEST_DIR, TIME_STEP)
    test_label = test_label.reshape(-1, 1)
    test_data = test_data[:, :, 0]

    # transfer training and validation data to Variable
    train_data, train_label = Variable(torch.from_numpy(train_data)).int(), Variable(torch.from_numpy(train_label)).float()
    val_data, val_label = Variable(torch.from_numpy(val_data)).int(), Variable(torch.from_numpy(val_label)).float()
    test_data, test_label = Variable(torch.from_numpy(test_data)).int(), Variable(torch.from_numpy(test_label)).float()

    # deploy the data into cuda device
    if torch.cuda.is_available():
        model = model.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        val_data = val_data.cuda()
        val_label = val_label.cuda()
        test_data = test_data.cuda()
        test_label = test_label.cuda()
    
    # build training dataset loader
    train_dataset = TensorDataset(train_data, train_label)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
    )

    # build validation dataset loader
    val_dataset = TensorDataset(val_data, val_label)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
    )

    # build testing dataset and loader
    test_dataset = TensorDataset(test_data, test_label)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
    )

    # training the model
    training_loss_iteration, validation_loss_iteration, iteration = training(model, train_loader, val_loader, test_loader)


    # plot the training history
    plot_training_history(training_loss_iteration, validation_loss_iteration, iteration)

    # plot the comparison for training data
    plot_comparison_pred_gt(model, train_dataset, train_label, "training")

    # plot the comparison for validation data
    plot_comparison_pred_gt(model, val_dataset, val_label, "validation")

    # plot the comparison for testing data 
    plot_comparison_pred_gt(model, test_dataset, test_label, "testing")



