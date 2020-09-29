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

#from __future__ import print_function, division

"""
resnet50 on CBIS-DDSM database:
This is an attempt to track performance in the article -
 https://www.mdpi.com/2313-433X/5/3/37/pdf


atract--
https://www.youtube.com/watch?v=yr9tQnlkDQw
"""


transform_imgs = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224, 0.225])
    ]),
}

data_dir = 'test_data'

img_data = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                    transform_imgs[x])
            for x in ['train','val']}

dataloaders = {x: torch.utils.data.DataLoader(img_data[x], batch_size=4,
                                              shuffle=True,
                                              num_workers=4)
               for x in ['train', 'val']}

data_size = {x: len(img_data[x]) for x in ['train','val']}

class_names = img_data['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")



def imshow(inp, title=None):
    """
    Imshow for Tensor
    :param inp:
    :param title:
    :return:
    """

    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp,0,1)
    plt.figure(figsize=(10,10))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)#--



def train_model(model, loss_fn, optimizer, scheduler, num_epochs =25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -1))
        print('-' * 10)

        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, outputs in dataloaders[phase]:
                inputs = inputs.to(device)
                outputs = outputs.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    probabilities = model(inputs)

                    _, predictions = torch.max(probabilities,1)
                    loss = loss_fn(probabilities,outputs)


                    if phase == 'train':
                        loss.backward()

                        optimizer.step()
                        running_loss += loss.item()* inputs.size(0)
                        running_corrects += torch.sum(predictions == outputs.data)

                    epoch_loss = running_loss / data_size[phase]
                    epoch_acc = running_corrects.double()/ data_size[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            #return the model with best model weights
            model.load_state_dict(best_model_wts)
            return model

if __name__ == "__main__":
    # inputs, output = next(iter(dataloaders['train']))
    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in output])
    #
    # output.data.numpy()


    """lol"""

    model_fun = models.resnet50(pretrained=True)

    for param in model_fun.parameters():
        param.requires_grad = False

    num_firs = model_fun.fc.in_features
    model_fun.fc = nn.Linear(num_firs, 2)

    #Sending Model to CUDA for training
    model_fun = model_fun.to(device)

    #Loss Function
    loss_fn = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = optim.SGD(model_fun.parameters(), lr=0.001, momentum = 0.5)

    #Decay Learning rate of optimizer by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model = train_model(model_fun, loss_fn, optimizer, exp_lr_scheduler, num_epochs=20)