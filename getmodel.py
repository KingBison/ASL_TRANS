
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from torch import optim
from torch.utils.data import DataLoader, TensorDataset

import pickle



def getModel():

    train_df = pd.read_csv('sign_mnist_train.csv').to_numpy()
    valid_df = pd.read_csv('sign_mnist_test.csv').to_numpy()

    x_train = train_df[:, 1:].reshape(-1, 1, 28, 28) / 255.0
    x_valid = valid_df[:, 1:].reshape(-1, 1, 28, 28) / 255.0

    y_train = train_df[:, :1].reshape(-1)
    y_valid = valid_df[:, :1].reshape(-1)

    #Create dataloaders
    train_ds = TensorDataset(torch.tensor(x_train, dtype = torch.float), torch.tensor(y_train, dtype = torch.long))
    train_dl = DataLoader(train_ds, batch_size=128, shuffle = True, num_workers = 4, drop_last = True)

    valid_ds = TensorDataset(torch.tensor(x_valid, dtype = torch.float), torch.tensor(y_valid, dtype = torch.long))
    valid_dl = DataLoader(valid_ds, batch_size=256, shuffle = True, num_workers = 4, drop_last = True)

    model = nn.Sequential(nn.Conv2d(1, 6, kernel_size = 5, padding=(2,2)), nn.Tanh(), 
                     nn.AvgPool2d(2), 
                     nn.Conv2d(6, 16 , kernel_size = 5), nn.Tanh(), 
                     nn.AvgPool2d(2), 
                     nn.Conv2d(16, 120, kernel_size=5), nn.Tanh(),
                     nn.Flatten(),
                     nn.Linear(120, 84), nn.Tanh(),
                     nn.Linear(84, 26))
    num_epochs = 25; lr = .8e-1;
    accuracies_cross_entropy = [] #We can use a higher learning rate with cross entroy loss. 
    losses = []

    opt = optim.SGD(model.parameters(), lr=lr)
    for i in range(num_epochs):
        for x, y in train_dl:
            yhat = model(x)
            loss = F.cross_entropy(yhat, y) #Takes care of softmax and one hot encoding for us!
            loss.backward()
            opt.step(); opt.zero_grad();
            losses.append(loss.item())
        
        #Check validation loss and accuracy at the end of each epoch:
        model.eval() #Put in evaluation mode!
        with torch.no_grad():
            x, y = next(iter(valid_dl)) #Just measure on one minibatch
            yhat = model(x)
            max_values, max_indices = torch.max(yhat, dim=1)
            accuracy = (max_indices.eq(y).sum().float()/len(y)).item()
            accuracies_cross_entropy.append(accuracy)
            print('Epoch: ' + str(i+1) + ', training loss = ' + str(round(loss.item(), 3)) + \
                ', valid accuracy = ' + str(round(accuracy, 3)))

    return model

if __name__ == '__main__':

    with open('model.model','wb') as f:

        pickle.dump(obj=getModel(),file=f,protocol=pickle.HIGHEST_PROTOCOL)