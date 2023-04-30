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


def retrieveVariables(filename):
    variables = []
    with open(str(filename), 'rb') as file:
        variables = pickle.load(file)
    return variables

# quick retrieve data
def retrieveVariables(filename):
    variables = []
    with open(str(filename), 'rb') as file:
        variables = pickle.load(file)
    return variables


# read in training data
def readin_data_train(datadir,readindata,batch_size):
    [envelop_signal, PmP_time, PmP_label, PmP_dist, PmP_evdp, PmP_mag, PmP_stlo, PmP_stla, PmP_evlo, PmP_evla, PmP_evid, PmP_fname] = retrieveVariables(f"{datadir}/{readindata}")

    print("size of data: ",len(envelop_signal))
    PmP_dist = (np.array(PmP_dist)-50)/150
    PmP_evdp = np.array(PmP_evdp)/20
    PmP_time = (np.array(PmP_time)-5)/25

    envelop_signal = preprocessing.normalize(envelop_signal, norm = 'max') 
    envelop_signal = preprocessing.scale(envelop_signal, axis = 0)
    
    prolong_length = 8 # originally was 8
    sample_num = len(envelop_signal)

    X = np.concatenate((envelop_signal, np.ones([sample_num, prolong_length])*PmP_dist[:,np.newaxis],\
                        np.ones([sample_num, prolong_length])*PmP_evdp[:,np.newaxis]), axis = 1)

    Y = np.concatenate((np.array(PmP_label)[:, np.newaxis], np.array(PmP_time)[:, np.newaxis]), axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X[:, np.newaxis], Y, test_size=.2, random_state=42)
    # create the training datset
    my_dataset1 = data.TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train).type(torch.float))
    train_loader = data.DataLoader(my_dataset1,batch_size=batch_size, shuffle=True)
    # create the testing datset
    my_dataset2 = data.TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test).type(torch.float))
    test_loader = data.DataLoader(my_dataset2,batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


# read in real data
def readin_data_real(datadir,readindata,batch_size):
    [envelop_signal, PmP_time, PmP_ptime, PmP_label, PmP_dist, PmP_evdp, PmP_mag, PmP_stlo, PmP_stla, PmP_evlo, PmP_evla, PmP_evid, PmP_fname] = retrieveVariables(f"{datadir}/{readindata}")

    PmP_dist = (np.array(PmP_dist)-50)/150
    PmP_evdp = np.array(PmP_evdp)/20
    PmP_time = (np.array(PmP_time)+abs(np.array(PmP_ptime))-5)/25

    envelop_signal = preprocessing.normalize(envelop_signal, norm = 'max') 
    envelop_signal = preprocessing.scale(envelop_signal, axis = 0)
    
    prolong_length = 8
    sample_num = len(envelop_signal)

    X = np.concatenate((envelop_signal, np.ones([sample_num, prolong_length])*PmP_dist[:,np.newaxis],\
                        np.ones([sample_num, prolong_length])*PmP_evdp[:,np.newaxis]), axis = 1)

    Y = np.concatenate((np.array(PmP_label)[:, np.newaxis], np.array(PmP_time)[:, np.newaxis]), axis=1)

    # create the testing datset
    my_dataset = data.TensorDataset(torch.Tensor(X[:, np.newaxis]),torch.Tensor(Y).type(torch.float))
    test_loader = data.DataLoader(my_dataset,batch_size=batch_size,shuffle=False)

    return test_loader


# Net structure
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
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
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DecodeBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(DecodeBlock, self).__init__()
        self.t_conv1 = nn.ConvTranspose1d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(True)
        self.t_conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.t_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.t_conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResnetEncoder(nn.Module):

    def __init__(self, block, decodeblock, num_blocks, num_outputs):
        self.inplanes = 128
        super(ResnetEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size = 2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(self.inplanes)


        self.avgpool = nn.AvgPool1d(kernel_size = 3, stride=3, padding=0)
        self.layer1 = self._make_layer(block, self.inplanes, 128, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, 128, num_blocks[1]) #, stride=2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=2, stride=2, padding=0, bias=False)

        self.decode_t_conv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4, padding = 1, bias=False)
        self.decode_bn1 = nn.BatchNorm1d(64)
        self.decode_layer1 = self._make_layer(decodeblock, 64, 64, num_blocks[0])
        self.decode_layer2 = self._make_layer(decodeblock, 64, 64, num_blocks[1])
        #self.decode_t_conv2 = nn.ConvTranspose1d(64, 64, kernel_size=4, stride=4, padding = 2, bias=False)
        self.decode_t_conv2 = nn.ConvTranspose1d(64, 64, kernel_size=13, stride=3, padding = 0, output_padding =0, bias=False)

        self.decode_bn2 = nn.BatchNorm1d(64)
        #self.decode_t_conv3 = nn.ConvTranspose1d(64, 1, kernel_size=3, stride=3, padding = 2, output_padding=1, bias=False)
        self.decode_t_conv3 = nn.ConvTranspose1d(64,1,kernel_size=12,stride=3,padding=0,output_padding=0, bias=False)
        
        self.predictor_conv1 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.predictor_bn1 = nn.BatchNorm1d(128)

        self.predictor_layer1 = self._make_layer(block, 128, 128, num_blocks[0])
        self.predictor_layer2 = self._make_layer(block, 128, 128, num_blocks[1])

        self.predictor_conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.predictor_bn2 = nn.BatchNorm1d(128)
        self.predictor_conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        #self.predictor_fc = nn.Linear(in_features=896, out_features=2)
        self.predictor_fc = nn.Linear(in_features=4608, out_features=2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
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

        # what is this forloop for?
        for i in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def predictor(self, x):
        x = self.predictor_conv1(x)
        x = self.predictor_bn1(x)
        x = self.relu(x)
        x = self.predictor_layer1(x)
        x = self.predictor_layer2(x)
        x = self.predictor_conv2(x)
        x = self.predictor_bn2(x)
        x = self.relu(x)
        x = self.predictor_conv3(x)
        x = x.view(x.size(0), -1)
        x = self.predictor_fc(x)

        return x

    def decode(self, x):
        #print("decode:")
        x = self.decode_t_conv1(x)
        #print(x.size())
        x = self.decode_bn1(x)
        #print(x.size())
        x = self.relu(x)
        x = self.decode_layer1(x)
        #print(x.size())
        x = self.decode_layer2(x) #why is there a second layer here, it was not shown in the paper
        #print(x.size())
        x = self.decode_t_conv2(x)
        #print(x.size())
        x = self.decode_bn2(x)
        x = self.relu(x)
        x = self.decode_t_conv3(x)
        #print(x.size())

        return x

    def encode(self, x):
        x = self.conv1(x)
        #print("encode:")
        #print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        #print(x.size())
        x = self.layer1(x)
        #print(x.size())
        x = self.avgpool(x)
        #print(x.size())
        x = self.layer2(x)
        #print(x.size())
        x = self.avgpool(x)
        #print(x.size())
        x = self.conv2(x)
        #print(x.size())
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        #print(x.size())
        x = self.conv3(x)
        #print(x.size())

        return x
    def forward(self, x):
        z = self.encode(x)
        z = self.predictor(z)
        return z


# Train PmPNet
def NetTrain(wdir,train_log,net_model,train_loader,learning_rate,num_epochs,batch_size,device):
    model = ResnetEncoder(BasicBlock, DecodeBlock, [2, 2], 2)
    if device.type == "cuda":
        model.cuda()
    
    f = open(f"{wdir}/{train_log}.txt","w")
    f.close()

    if device.type == "cuda":
        class_weights = torch.FloatTensor([20.]).cuda()
    else:
        class_weights = torch.FloatTensor([20.])
    criterion = nn.MSELoss(reduction='mean')
    PmPcriterion = torch.nn.BCEWithLogitsLoss(pos_weight = class_weights)
    traveltimecriterion = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            latents = model.encode(signals)
            prediction = model(signals)
            outputs = model.decode(latents)
            #loss1: reconstruction loss for autoencoder
            loss1 = criterion(outputs, signals)
            #loss2: PmP label prediction loss, binary cross entropy loss
            loss2 = PmPcriterion(prediction[:,0], labels[:,0])
            #loss3: travel time loss, l1 loss
            loss3 = traveltimecriterion(prediction[:, 1], labels[:, 1])

            loss = loss1 + loss2 + loss3

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 1:
                print ("Epoch [{}/{}], Step [{}/{}] Loss1: {:.6f},Loss2: {:.6f},Loss3: {:.6f}"
                       .format(epoch+1, num_epochs, i+1, total_step, loss1.item(),loss2.item(),loss3.item()))
                f = open(f"{wdir}/{train_log}.txt","a") 
                f.write("Bach_Size %d Epoch %d %d Step %d %d Loss %.4f Loss1 %.4f Loss2 %.4f Loss3 %.4f\n" % (batch_size, epoch+1, num_epochs, i+1, total_step, loss.item(), loss1.item(), loss2.item(), loss3.item()))
                f.close()
        #Decay learning rate
        if (epoch+1) % 10 == 0:
            curr_lr /= 2
            update_lr(optimizer, curr_lr)
            
    torch.save(model.state_dict(), f"{wdir}/{net_model}")


# model evaluation on test data
def pr_curve(p_threshold, model, test_data, dev):
    m = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        TP = np.zeros(len(p_threshold))
        FP = np.zeros(len(p_threshold))
        TN = np.zeros(len(p_threshold))
        FN = np.zeros(len(p_threshold))
        for images, labels in test_data:
            images = images.to(dev)
            labels = labels.to(dev)
            latents = model.encode(images)
            predicted_prob = m(model.predictor(latents)[:, 0])
            print(model.predictor(latents)[:, 0])
            for i in range(len(labels)):
                for j in range(len(p_threshold)):
                    if predicted_prob[i] > p_threshold[j]:
                        if labels[i,0] == 1:
                            TP[j] +=1
                        else:
                            FP[j] +=1
                    else:
                        if labels[i,0] ==1:
                            FN[j] +=1
                        else:
                            TN[j] +=1
    precision = (TP+0.01)/(TP+FP+0.01)
    recall = (TP+0.01)/(TP+FN+0.01)
    return [precision, recall]



def netevalu(wdir,net_model,prcurve_file,predict_file,test_loader,device):
    model = ResnetEncoder(BasicBlock, DecodeBlock, [2, 2], 2)
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

            predicted_traveltime = torch.cat([predicted_traveltime, prediction[:, 1]])
            true_traveltime = torch.cat([true_traveltime, labels[:, 1]])
            
    f = open(f"{wdir}/{predict_file}.txt","w")
    Nump = np.arange(0,len(predicted_traveltime),1)
    for n in Nump:
        f.write("predicted_traveltime %.4f true_traveltime %.4f\n" % (predicted_traveltime[n]*25+5,true_traveltime[n]*25+5))
    f.close()

    p_threshold = np.array([1e-9, 1e-2, 1e-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.9999999])
    pr_curve2 = np.asarray(pr_curve(p_threshold, model, test_loader, device))
    f = open(f"{wdir}/{prcurve_file}.txt","w")
    Nump = np.arange(0,len(p_threshold),1)
    for n in Nump:
        f.write("Precision %.4f Recall %.4f pthresh %.4f\n" % (pr_curve2[0][n],pr_curve2[1][n],p_threshold[n]))
    f.close()


# Identify PmP phase on real set
def netpredict(datadir,readindata,wdir,net_model,predict_file,test_loader,device):
    model = ResnetEncoder(BasicBlock, DecodeBlock, [2, 2], 2)
    model.load_state_dict(torch.load(f"{wdir}/{net_model}"))
    model.cuda()
    m = nn.Sigmoid()

    [envelop_signal, PmP_time, PmP_ptime, PmP_label, PmP_dist, PmP_evdp, PmP_mag, PmP_stlo, PmP_stla, PmP_evlo, PmP_evla, PmP_evid, PmP_fname] = retrieveVariables(f"{datadir}/{readindata}")

    predicted_PmPlabel = torch.tensor([]).cuda()
    predicted_traveltime = torch.tensor([]).cuda()
    model.eval()
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)
            prediction = model(signals)
            probability = m(prediction[:,0])

            predicted_PmPlabel = torch.cat([predicted_PmPlabel, probability])
            predicted_traveltime = torch.cat([predicted_traveltime, prediction[:, 1]])

    f=open(f"{wdir}/{predict_file}",'w')
    i=0
    while(i<len(envelop_signal)):
        print("NO.: %d   ID: %d   PmP_Prob: %f  PmP_Time: %f  dist: %.1f   evdp: %.2f   mag: %.1f  evtnm: %s" %(i,PmP_evid[i],predicted_PmPlabel[i],predicted_traveltime[i]*25+5,PmP_dist[i],PmP_evdp[i],PmP_mag[i],PmP_fname[i]))
        f.write("%d  %d  %f  %.1f  %.2f  %.2f  %.1f  %s\n" %(i,PmP_evid[i],predicted_PmPlabel[i],predicted_traveltime[i]*25+5,PmP_dist[i],PmP_evdp[i],PmP_mag[i],PmP_fname[i]))
        i=i+1
    f.close()


# plot the model evaluation
def plot_modeva(wdir,train_log,prcurve_file,predict_file,plot_fname):
    f = plt.figure(figsize=(20, 3.5))
    grid = plt.GridSpec(1, 4, wspace=0.4, hspace=0.3)
    font2 = {'weight' : 'normal','size' : 18,}
    rate = 4
    ax1 = f.add_subplot(grid[0, 0:2])
    data = pd.read_csv(f"{wdir}/{train_log}.txt", sep=" ", header=None)
    data.columns = ["Term1", "batchsize", "Term2", "epoch", "epochAll", "Term3", "step1", "step2", "Term4", "residual", "Term5", "residual1", "Term6", "residual2", "Term7", "residual3"]
    ax1.semilogy(data['epoch'][np.arange(0,len(data['epoch']),rate)], data['residual'][np.arange(0,len(data['epoch']),rate)], "o", markersize=6, color = "black")
    ax1.set_xlabel('Epoch',font2)
    ax1.set_ylabel('Loss',font2)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2) # the previous version was b = False
    ax1.set_ylim(0.5*min(data['residual']),1.2*max(data['residual']),)
    ax1.yaxis.grid(True)
    ax1.set_xlim(-0.05,0.05+80)
    ax1.set_xticks(np.arange(0,80+0.001,10))
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ###################################################
    ax2 = f.add_subplot(grid[0, 2:3])
    data = pd.read_csv(f"{wdir}/{prcurve_file}.txt", sep=" ", header=None)
    data.columns = ["Term1","precision", "Term2", "recall", "Term3", "probability"]
    ax2.plot(data['precision'], data['recall'], color = "black", linewidth=6)
    ax2.plot(data['precision'], data['recall'], "o", markersize=2, color = "white", clip_on=False)
    ax2.set_xlabel('Precision',font2)
    ax2.set_ylabel('Recall',font2)
    ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax2.yaxis.grid(True)
    ax2.xaxis.grid(True)
    ax2.set_xlim(-0.05,1+0.05)
    ax2.set_xticks(np.arange(0,1+0.001,0.2))
    ax2.set_ylim(-0.05,1+0.05)
    ax2.set_yticks(np.arange(0,1+0.001,0.2))
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ###################################################
    ax3 = f.add_subplot(grid[0, 3:4])
    data = pd.read_csv(f"{wdir}/{predict_file}.txt", sep=" ", header=None)
    data.columns = ["Term1","pdt_time", "Term2", "obs_time"]
    ax3.hist(x=(data['obs_time']-data['pdt_time']), bins=100, color='black',alpha=1.0, rwidth=0.85)
    ax3.set_xlabel('Traveltime Residual (s)',font2)
    ax3.set_ylabel('Count',font2)
    ax3.set_xlim(-2.0,2.0+0.001)
    ax3.set_xticks(np.arange(-2.0,2.0+0.001,1))
    ax3.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    ax3.xaxis.grid(True)
    plt.grid(visible=True, which='minor')
    ax3.tick_params(axis='both', which='major', labelsize=14)
    plt.savefig(f"{wdir}/{plot_fname}.png", bbox_inches='tight', dpi=300)
    plt.show()


# plot PmPNet's prediction
def plot_modpredict(wdir,predict_file,plot_fname):
    df = pd.read_csv(f"{wdir}/{predict_file}",sep="  ", header=None, engine='python')
    df.columns = ['NO', 'ID', 'Prob', 'Time', 'dist', 'evdp', 'mag', 'evtnm']

    f = plt.figure(figsize=(18, 2.5))
    grid = plt.GridSpec(1, 6, wspace=0.6, hspace=0.3)
    ax1 = f.add_subplot(grid[0, 0:2])
    ax1.plot(range(0,len(df["Prob"])),df["Prob"], "o", markersize=0.1, color = "red")
    ax1.set_xlabel('Waveform_No.')
    ax1.set_ylabel('Probability')
    #########################################################################################
    ax2 = f.add_subplot(grid[0, 2:4])
    ax2.hist(df["Prob"], bins =60, range=(0,1), color = "red", histtype="bar", edgecolor = 'snow')
    ax2.set_xlim((0,1))
    ax2.set_xticks(np.arange(0,1.1,0.1))
    ax2.set_xlabel('PmP Probability')
    ax2.set_ylabel('Counts')
    ###########################################################################################
    ax3 = f.add_subplot(grid[0, 4:6])
    ax3.hist(df["Time"], bins =60, range=(0,35), color = "red", histtype="bar", edgecolor = 'snow')
    ax3.set_xlim((5,35))
    ax3.set_xticks(np.arange(5,35.1,5))
    ax3.set_xlabel('Traveltime of PmP')
    ax3.set_ylabel('Counts')

    plt.savefig(f"{wdir}/{plot_fname}.png", bbox_inches='tight', dpi=300)
    plt.show()