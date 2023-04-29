## 0313

from loaddata0216 import Loaddata

import json
import click as ck
import pandas as pd
from torch_utils import FastTensorDataLoader
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

import torch.utils.data as Data
from torchvision.models import *
import time
import math
import random
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

import os
@ck.command()

@ck.option(
    '--data-root', '-dr', default='data',
    help='ppi and interactions folder root')
@ck.option(
    '--negativefold', '-nf', default=4,
    help='negativefold , negatives is x fold of positives') 
@ck.option(
    '--name', '-n', default='currname',
    help='names of file') 
@ck.option(
    '--esmfolder', '-esmf',
    help='esmfolder of esm2') 
@ck.option(
    '--modelpath', '-lp',default = '',
    help='modelpath for loading the model to do the prediction') 
@ck.option(
    '--svpath', '-sv',default = '',
    help='sv the embedding for all proteins') 


def main(modelpath,svpath,data_root,name,esmfolder,negativefold):
    batch_size =128
    net = twoFeaturedotmodel(in_dim=1280,hidden_dim=100,out_dim=1280)
    
    print('>>>model setting done\n',flush=True)

    print('>>>data preparing..\n',flush=True)

    name2index_file = f'{data_root}/{name}_name2index.pkl'
    index2name_file = f'{data_root}/{name}_index2name.pkl'
    esmfolder = esmfolder
    svfolder = f'{data_root}/'
    allinfo_file = f'{data_root}/{name}_allinfo.pkl'
    allpairs_file = f'{data_root}/{name}_allpairs.pkl'
    
    allpairs_csvfile = f'{data_root}/{name}_allpairs.csv'
    if not os.path.exists(allpairs_csvfile):
        print('>>>!!not exists',flush=True)
        dataset = Loaddata(high_ppipath,low_ppipath,esmfolder,svfolder,negativefold,name)
        allinfo = dataset.allinfo 
        allpairs = dataset.allpairs
        name2index = dataset.name2index
        index2name = dataset.index2name
        print('>>>sv...',flush=True)
        print('>>>finished sv',flush=True)

    else:

        print('\n\n >>>load...',flush=True)
        allinfo = pd.read_pickle(allinfo_file)
        allpairs = pd.read_csv(allpairs_csvfile)
        
        allpairs = allpairs.astype(int)
        with open(name2index_file,'r')as f:
            name2index = json.load(f)

    model = 'twof'
    
    predallinfo_file = f'{svpath}/{name}_predicted_{model}_allinfo.pkl'
    print('filepath=',predallinfo_file,flush=True)
    

    predata = get_testdata(allinfo)
    pred_loader = FastTensorDataLoader(
            *predata, batch_size=batch_size, shuffle=True)
    print(' # data prepared #',flush=True)

    print('..Loading model',flush=True)
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    dd ={}
    with torch.no_grad():
        test_steps = int(math.ceil(len(predata[1]) / batch_size))
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features,bathch_indexs in pred_loader:
                bar.update(1)

                batch_features = net.predict(batch_features)
                bathch_indexs = torch.squeeze(bathch_indexs)
                # print ('bathch_indexs:\n',bathch_indexs.tolist())
                # print ('batch_features:\n',batch_features)
                # print('batch_f=\n',batch_features)
                for i in range(0,len(bathch_indexs)) :
                    ind = bathch_indexs[i]
                    try:
                        f1 = batch_features[0][i].detach().numpy().tolist()
                        f2 = batch_features[1][i].detach().numpy().tolist()
                        dd[int(ind)] = [f1,f2]
                    except:
                        print('int(ind)={}=batch_index'.format(int(ind)))
                        print('len feature={}'.format(len(batch_features)))
                        print('len index={}'.format(len(bathch_indexs)))

        print('..sv info',flush=True)

        predallinfo = pd.DataFrame(dd.items(),columns=['index', 'predf1_f2'])
        predallinfo.to_pickle(predallinfo_file)
    print('..sv done',flush=True)
    

def get_testdata(df):
    a = df['feature'].values
    a = np.vstack(a)
    ff = torch.from_numpy(a).float()

    a = df['index'].values
    a = np.vstack(a)
    ii = torch.from_numpy(a).float()
    return ff,ii

def get_data(df,pairs, newpairsflage = False):
    data = torch.zeros((len(pairs), 2560), dtype=torch.float32)
    labels = torch.zeros((len(pairs), 1), dtype=torch.float32)
    newpairs = torch.zeros((len(pairs), 2), dtype=torch.float32)
    for i, row in enumerate(pairs.itertuples()):
        try:
            f1 = df.loc[df['index']==row.p1].feature.values.tolist()[0]
            f2 = df.loc[df['index']==row.p2].feature.values.tolist()[0]
            f1 = torch.tensor(f1)
            f2 = torch.tensor(f2)

        except:
            # print('index p1 =',row.p1)
            # print('index p2 =',row.p2,flush=True)
            continue

        newf = torch.cat((f1,f2),0)
        i1 = torch.tensor(row.p1,dtype=torch.float32).unsqueeze(axis=0)
        i2 = torch.tensor(row.p2,dtype=torch.float32).unsqueeze(axis=0)
        newp = torch.cat((i1,i2),0) 
        # ff1,ff2 = newf.chunk(2)
        data[i,:] = torch.FloatTensor(newf)
        labels[i,:] = torch.tensor(row.label)
        newpairs[i,:] = torch.tensor(newp)
    if newpairsflage ==True:
        return data,labels,newpairs
    else:
        return data,labels

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

class Featuredotmodel(nn.Module):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        output = self.Sigmoid(dotproduct)
        return output

    def predict(self,x:torch.Tensor):
        # x0 here is the esm2 embedding 
        #return the 
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        return x0

class twoFeaturedotmodel(nn.Module):
    #twof with relu works
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.linear3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear4 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        print('relu as activation between layers',flush=True)
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        # x0 = self.Sigmoid(x0)
        x0 = self.relu(x0)
        x0 = self.linear2(x0)

        x1 = self.linear3(x1)
        # x1 = self.Sigmoid(x1)
        x1 = self.relu(x1)
        x1 = self.linear4(x1)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        # print('dotproduct',dotproduct.size(),flush=True)
        output = self.Sigmoid(dotproduct)
        return output

    def predict(self,x:torch.Tensor)-> torch.Tensor:

        
        x0 = self.linear1(x)
        # x0 = self.Sigmoid(x0)
        x0 = self.relu(x0)
        x0 = self.linear2(x0)

        x1 = self.linear3(x)
        # x1 = self.Sigmoid(x1)
        x1 = self.relu(x1)
        x1 = self.linear4(x1)
        return x0,x1

if __name__ == '__main__':
    main()