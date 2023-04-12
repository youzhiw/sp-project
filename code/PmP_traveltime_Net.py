### Developed by Wen Ding and Tianjue Li, 2022 ###
# import needed mudlues 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import pickle

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.utils import data

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# quick retrieve data
def retrieveVariables(filename):
    variables = []
    with open(str(filename), 'rb') as file:
        variables = pickle.load(file)
    return variables


# read in training data 
def readin_data_train(datadir,readindata,batch_size):
    envelop_signal, PmP_time, PmP_ptime, PmP_label, PmP_dist, PmP_evdp, PmP_mag, *_= retrieveVariables(f"{datadir}/{readindata}")
    # data normalization
    PmP_dist = (np.array(PmP_dist)-50)/150
    PmP_evdp = np.array(PmP_evdp)/20
    PmP_time = np.array(PmP_time)
    envelop_signal = preprocessing.normalize(envelop_signal, norm = 'max')
    envelop_signal = preprocessing.scale(envelop_signal, axis = 0)

    prolong_length = 8
    sample_num = len(envelop_signal)
    X = np.concatenate((envelop_signal, np.ones([sample_num, prolong_length])*PmP_dist[:,np.newaxis],\
                        np.ones([sample_num, prolong_length])*PmP_evdp[:,np.newaxis]), axis = 1)[:, np.newaxis]
    Y = PmP_time[:, np.newaxis]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=42)
    # create the training datset
    my_dataset1 = data.TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train).type(torch.float))
    train_loader = data.DataLoader(my_dataset1,batch_size=batch_size, shuffle=True)
    # create the testing datset
    my_dataset2 = data.TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test).type(torch.float))
    test_loader = data.DataLoader(my_dataset2,batch_size=batch_size, shuffle=True)

    return train_loader,test_loader


# read in real data
def readin_data_real(datadir,readindata,batch_size):
    envelop_signal, PmP_time, PmP_ptime, PmP_label, PmP_dist, PmP_evdp, PmP_mag, PmP_stlo, PmP_stla, PmP_evlo, PmP_evla, PmP_evid, PmP_fname= retrieveVariables(f"{datadir}/{readindata}")

    PmP_dist = (np.array(PmP_dist)-50)/150
    PmP_evdp = np.array(PmP_evdp)/20
    PmP_time = np.array(PmP_time)
    envelop_signal = preprocessing.normalize(envelop_signal, 'max')
    evelop_signal = preprocessing.scale(envelop_signal, axis=0)

    prolong_length = 8
    sample_num = len(evelop_signal)
 
    X_test = np.concatenate((evelop_signal, np.ones([sample_num, prolong_length])*PmP_dist[:,np.newaxis],\
                        np.ones([sample_num, prolong_length])*PmP_evdp[:,np.newaxis]), axis = 1)[:, np.newaxis]
    Y_test = PmP_time[:, np.newaxis]

    test_loader = data.DataLoader(data.TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test).type(torch.float)),batch_size=batch_size)

    return test_loader

# Net Structure
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.activation = nn.Mish()
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out

nn_size = 128
class Resnet(nn.Module):
    
    def __init__(self, basicblock, num_blocks):
        self.inplanes = nn_size
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv1d(nn_size, nn_size, kernel_size=2, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(nn_size)
        self.avgpool = nn.AvgPool1d(kernel_size = 2, stride=2, padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride=2, padding=0)
        self.layer = self._make_layer(basicblock, self.inplanes, nn_size, num_blocks)
        self.conv3 = nn.Conv1d(nn_size, nn_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(nn_size)
        self.fc = nn.Linear(in_features=9*nn_size, out_features=1)
  
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        
        for i in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = self.layer(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Train PmP-traveltime-Net
def NetTrain(wdir,train_log,net_model,train_loader,learning_rate,num_epochs,batch_size,device):
    model = Resnet(BasicBlock,10)
    model.cuda()

    f = open(f"{wdir}/{train_log}.txt","w")
    f.close()

    l1 = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    # For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate
    model.train()
    model.zero_grad()
    for epoch in range(num_epochs):
        for i, (signals, labels) in enumerate(train_loader):
            signals = signals.to(device)
            labels = labels.to(device)
            prediction = model(signals)
            loss = l1(prediction, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 50 == 1:
                print ("Epoch [{}/{}],  Step [{}/{}],  Loss: {:.6f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                f = open(f"{wdir}/{train_log}.txt","a") 
                f.write("Bach_Size %d Epoch %d %d Step %d %d Loss %.4f\n" % (batch_size, epoch+1, num_epochs, i+1, total_step, loss.item()))
                f.close()
        #Decay learning rate
        if (epoch+1) % 3 == 0:
            curr_lr /= 1.1
            update_lr(optimizer, curr_lr)

    torch.save(model.state_dict(), f"{wdir}/{net_model}")


# model evaluation on test set
def netevalu(wdir,net_model,predict_file,test_loader,device):
    model = Resnet(BasicBlock,10)
    model.load_state_dict(torch.load(f"{wdir}/{net_model}"))
    model.cuda()

    predicted_traveltime = torch.tensor([]).cuda()
    true_traveltime = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            prediction = model(signals)
            predicted_traveltime = torch.cat([predicted_traveltime, prediction])
            true_traveltime = torch.cat([true_traveltime, labels])

    f = open(f"{wdir}/{predict_file}.txt","w")
    Nump = np.arange(0,len(predicted_traveltime),1)
    for n in Nump:
        f.write("predicted_traveltime %.4f true_traveltime %.4f\n" % (predicted_traveltime[n],true_traveltime[n]))
    f.close()


# predict the PmP traveltime on real set
def netpredict(datadir,readindata,wdir,net_model,predict_file,test_loader,device):
    model = Resnet(BasicBlock,10)
    model.load_state_dict(torch.load(f"{wdir}/{net_model}"))
    model.cuda()

    envelop_signal, PmP_time, PmP_ptime, PmP_label, PmP_dist, PmP_evdp, PmP_mag, PmP_stlo, PmP_stla, PmP_evlo, PmP_evla, PmP_evid, PmP_fname= retrieveVariables(f"{datadir}/{readindata}")

    predicted_traveltime = torch.tensor([]).cuda()
    true_traveltime = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            prediction = model(signals)
            predicted_traveltime = torch.cat([predicted_traveltime, prediction])
            true_traveltime = torch.cat([true_traveltime, labels])

    f = open(f"{wdir}/{predict_file}.txt","w")
    Nump = np.arange(0,len(predicted_traveltime),1)
    for n in Nump:
        f.write("predicted_traveltime %.4f true_traveltime %.4f file_name %s dist %.1f evdp %.1f mag %.1f\n" % (predicted_traveltime[n]+abs(PmP_ptime[n]),true_traveltime[n]+abs(PmP_ptime[n]),PmP_fname[n],PmP_dist[n],PmP_evdp[n],PmP_mag[n]))
    f.close()


# plot the model evaluation
def plot_modeva(wdir,train_log,predict_file,plot_fname):
    f = plt.figure(figsize=(20, 3.5))
    grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
    font2 = {'weight' : 'normal','size' : 18,}
    rate = 1
    ax1 = f.add_subplot(grid[0, 0:2]);
    data = pd.read_csv(f"{wdir}/{train_log}.txt", sep=" ", header=None)
    data.columns = ["Term1", "batchsize", "Term2", "epoch", "epochAll", "Term3", "step1", "step2", "Term4", "residual"]
    ax1.semilogy(data['epoch'][np.arange(0,len(data['epoch']),rate)], data['residual'][np.arange(0,len(data['epoch']),rate)], "o", markersize=3, color = "black")
    ax1.set_xlabel('Epoch',font2)
    ax1.set_ylabel('Loss',font2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax1.yaxis.grid(True)
    ax1.set_xlim(-0.05,0.05+200)
    ax1.set_xticks(np.arange(0,200+0.001,20))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ###################################################
    ax2 = f.add_subplot(grid[0, 2:4])
    data = pd.read_csv(f"{wdir}/{predict_file}.txt", sep=" ", header=None)
    data.columns = ["Term1","pdt_time", "Term2", "obs_time"]
    ax2.hist(x=(data['obs_time']-data['pdt_time']), bins=100, color='black',alpha=1.0, rwidth=0.85)
    ax2.set_xlabel('Traveltime Residual (s)',font2)
    ax2.set_ylabel('Count',font2)
    ax2.set_xlim(-2.0,2.0+0.001)
    ax2.set_xticks(np.arange(-2.0,2.0+0.001,0.5))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax2.xaxis.grid(True)
    plt.grid(b=True, which='minor')
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(f"{wdir}/{plot_fname}.png", bbox_inches='tight', dpi=300)
    plt.show()


# plot the comparison between predicted and manually picked PmP travletimes
def plot_predict_compare(wdir,predict_file,plot_fname):
    f = plt.figure(figsize=(20, 3.5))
    grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
    font2 = {'weight' : 'normal','size' : 18,}
    ax1 = f.add_subplot(grid[0, 0:2])
    data = pd.read_csv(f"{wdir}/{predict_file}.txt", sep=" ", header=None)
    data.columns = ["Term1","pdt_time","Term2","obs_time","Term3","file_name","Term4","dist","Term5","evdp","Term6","mag"]
    ax1.plot(data['dist'], data['pdt_time'], "o", markersize=3, color = "red", label="Predicted")
    ax1.plot(data['dist'], data['obs_time'], "o", markersize=3, color = "blue", label="Observed")
    ax1.legend()
    ax1.set_xlabel('Distance (km)',font2)
    ax1.set_ylabel('PmP Traveltime (s)',font2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax1.yaxis.grid(True)
    ax1.set_xlim(40,210+0.001)
    ax1.set_xticks(np.arange(40,210+0.001,20))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ###################################################
    ax2 = f.add_subplot(grid[0, 2:4])
    ax2.hist(x=(data['obs_time']-data['pdt_time']), bins=100, color='black',alpha=1.0, rwidth=0.85)
    ax2.set_xlabel('Traveltime Residual (s)',font2)
    ax2.set_ylabel('Count',font2)
    ax2.set_xlim(-2.0,2.0+0.001)
    ax2.set_xticks(np.arange(-2.0,2.0+0.001,0.5))
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax2.xaxis.grid(True)
    plt.grid(b=True, which='minor')
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(f"{wdir}/{plot_fname}.png", bbox_inches='tight', dpi=300)
    plt.show()


# plot the net-predicted PmP travletimes
def plot_predict(wdir,predict_file,plot_fname):
    f = plt.figure(figsize=(10, 3.5))
    grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
    font2 = {'weight' : 'normal','size' : 18,}
    ax1 = f.add_subplot(grid[0, 0:4])
    data = pd.read_csv(f"{wdir}/{predict_file}.txt", sep=" ", header=None)
    data.columns = ["Term1","pdt_time","Term2","obs_time","Term3","file_name","Term4","dist","Term5","evdp","Term6","mag"]
    ax1.plot(data['dist'], data['pdt_time'], "o", markersize=3, color = "black")
    ax1.set_xlabel('Distance (km)',font2)
    ax1.set_ylabel('PmP Traveltime (s)',font2)
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax1.yaxis.grid(True)
    ax1.set_xlim(40,210+0.001)
    ax1.set_xticks(np.arange(40,210+0.001,20))
    ax1.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig(f"{wdir}/{plot_fname}.png", bbox_inches='tight', dpi=300)
    plt.show()