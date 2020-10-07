import time
import os
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

# from __future__ import print_function, division

"""
resnet50 on CBIS-DDSM database:
This is an attempt to track performance in the article -
 https://www.mdpi.com/2313-433X/5/3/37/pdf

"""

# Image augmentation - to avoid overfitting.
transform_imgs = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # to rgb images,the secend list is to standard deviation
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),

}

# connect to the data directory
data_dir = "//breast_dataset"  #############################################################################training and val
# two dictionary, one to train and val and second is to cancer and not cancer
img_data = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                    transform_imgs[x])
            for x in ['train', 'val', 'test']}
# make batch
dataloaders = {x: torch.utils.data.DataLoader(img_data[x], batch_size=32,
                                              shuffle=True,
                                              num_workers=4)
               for x in ['train', 'val', 'test']}

data_size = {x: len(img_data[x]) for x in ['train', 'val', 'test']}

class_names = img_data['train'].classes
# run on the gpu
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")


def imshow(inp, title=None):
    """
    Imshow for Tensor
    :param inp:
    :param title:
    :return:
    """

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # --


def train_model(model, loss_fn, optimizer, scheduler, num_epochs=32):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # create a copy of the pretrained model weights
    # and keep updating as training goes on
    best_acc = 0.0

    for epoch in range(num_epochs):  # for number of epochs
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evalate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, outputs in dataloaders[phase]:
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                optimizer.zero_grad()  # zero the parameter gradients

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    probabilities = model(inputs)  # Get the probabilities for set of images

                    _, predictions = torch.max(probabilities,
                                               1)  # Get the prediction for each batch image based on the maximun probability.

                    loss = loss_fn(probabilities, outputs)

                    if phase == 'train':  # backward and update paramaters of model
                        loss.backward()

                        optimizer.step()
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(predictions == outputs.data)

                    epoch_loss = running_loss / data_size[phase]
                    epoch_acc = running_corrects.double() / data_size[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    # update the model weights if model validation accuracy is better than before
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # return the model with best model weights
            model.load_state_dict(best_model_wts)
            return model


def testing(model):

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data
            images = images.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            predicted = predicted.to('cpu').numpy()
            correct += (predicted == labels.numpy()).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    # to show the images we add to the model
    # inputs, output = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in output])
    #
    # output.data.numpy()

    """connect the resnet50 model of pytorch with new train with the new images"""

    model_fun = models.resnet50(pretrained=True)

    for param in model_fun.parameters():
        param.requires_grad = False

    num_firs = model_fun.fc.in_features
    model_fun.fc = nn.Linear(num_firs, 2)

    # Sending Model to CUDA for training
    model_fun = model_fun.to(device)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model_fun.parameters(), lr=0.001, momentum=0.5)

    # Decay Learning rate of optimizer by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # call to the function of the train with the resnet model
    model = train_model(model_fun, loss_fn, optimizer, exp_lr_scheduler, num_epochs=20)

    # save model
    PATH = "entire_model.pt"
    torch.save(model, PATH)

    # Load
    model = torch.load(PATH)
    model.eval()
    testing(model)
