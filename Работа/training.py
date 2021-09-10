import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
from data import *
import numpy as np

def check_model(model, dataloader, data, device):
    was_training = model.training
    model.eval()
    correct = np.zeros(len(dataloader) * dataloader.batch_size, dtype=np.bool_)
    i = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            #print ("l",labels)
            #print("p", predictions)
            #print("d",data.noise_or_not[i:i+64])
            for label, prediction in zip(labels, predictions):
                answ = (label == prediction)
                if answ:
                    if data.noise_or_not[i]:
                        TP += 1
                    else:
                        FP += 1

                else:
                    if data.noise_or_not[i]:
                        FN += 1
                    else:
                        TN += 1
                correct[i] = answ
                i += 1
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2 * (recall * precision) / (recall + precision)
    print("acc: ", TP + TN, (TP + TN) / i)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)
    model.train(mode=was_training)
    return (TP + TN) / i, precision, recall, F1




def train_fixed_feature_extractor(model, dataloader, device, params):
    model_conv = model
    classes = params.num_classes
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, classes)

    model = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer = optim.SGD(model.fc.parameters(), lr=params.initial_lr, momentum=params.momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=params.step_size, gamma=params.gamma)
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloader, params.epochs, device,
                           params.batch_size)
    return model_ft


def train_model(model, criterion, optimizer, scheduler, dataloader, num_epochs, device, batch_size):
    since = time.time()
    dataset_size = len(dataloader) * batch_size
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        phases = ['train']

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloader:
                labels = labels.type(torch.LongTensor)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients - not done in baseline mode
                if optimizer:
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds_noise = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds_noise == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model
