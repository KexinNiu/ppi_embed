from loaddata0216 import Loaddata

import json
import click as ck
import pandas as pd
from torch_utils import FastTensorDataLoader
import torch
import numpy
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
from sklearn.metrics import roc_curve, auc, matthews_corrcoef,precision_recall_curve
from utils import compute_rank,compute_rank_roc,compute_roc,compute_MAP

import os
@ck.command()

@ck.option(
    '--data-root', '-dr', default='data',
    help='ppi and interactions folder root')
@ck.option(
    '--batch-size', '-bs', default=16,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
@ck.option(
    '--hiddim', '-hd', default=100,
    help='hidden dimension of the model') 
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
    '--modelname', '-mn',
    help='name of model,onef,twof')
@ck.option(
    '--metric', '-metric',
    default = 'dot',
    help='options =')
    # ['dot','Euclidean distance','Manhattan metric','Chebychev metric']
    





# data_root ="/home/niuk0a/Documents/COOOODE/simembed/new/"  
# high_ppipath='/home/niuk0a/Documents/COOOODE/simembed/4932.highconf.links.txt'
# low_ppipath='/home/niuk0a/Documents/COOOODE/simembed/4932.lowconf.links.txt'
# esmfolder='xxxxxxrandom gene'
# svfolder='/home/niuk0a/Documents/COOOODE/simembed/new/'


def main(data_root, batch_size, epochs, hiddim, load, device, negativefold,name,esmfolder, modelname,metric):
    print('>>>',torch.cuda.is_available())
    print('batch_size=',batch_size,flush=True)

    model_file = f'{data_root}/model_esm2_{epochs}_{modelname}_{metric}_{batch_size}.th'
    # # loadmodelfile
    # if load:

    # out_file = f'{data_root}/predictions_{epochs}_{modelname}_{metric}_{batch_size}.pkl'

    allinfo_file = f'{data_root}/{name}_allinfo.pkl'
    allpairs_file = f'{data_root}/{name}_allpairs.pkl'
    
    allpairs_csvfile = f'{data_root}/{name}_allpairs.csv'

    name2index_file = f'{data_root}/{name}_name2index.pkl'
    index2name_file = f'{data_root}/{name}_index2name.pkl'
    high_ppipath = f'{data_root}/{name}.highconf.links.txt'
    low_ppipath = f'{data_root}/{name}.lowconf.links.txt'
    esmfolder = esmfolder
    svfolder = f'{data_root}/'
    

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
        # allpairs = pd.read_pickle(allpairs_file)
        allpairs = pd.read_csv(allpairs_csvfile)
        # print('>>>allpairs top5\n',allpairs[:2],flush=True)
        allpairs = allpairs.astype(int)
        with open(name2index_file,'r')as f:
            name2index = json.load(f)

        print('>>>finished loading\n',flush=True)

    if modelname=='onef':
        if metric =='dot':
            net = Featuredotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280)
        

    elif modelname =='twof':
        if metric =='dot':
            net = twoFeaturedotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280)
        
    loss_func = nn.BCELoss()
    net.eval()

    # lr = lr
    # print('lr=',lr,flush=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)
    print('>>>model setting done\n',flush=True)

    
    # model 
    ########################################

    kf = KFold(n_splits = 5, shuffle = True, random_state = 17)
    for i, (train_index, test_index) in enumerate(kf.split(allinfo)):
        print(f"Fold {i}:",flush=True)
        # if i ==1:
        #     break
     
        test_data= allinfo.iloc[test_index]
        test_pairs = allpairs[(allpairs['p1'].isin(test_index)) & (allpairs['p2'].isin(test_index))]
        print(' # data prepare##',flush=True)
        #####################################################

        # Loading best model
        print('..Loading the best model')
        net.load_state_dict(torch.load(model_file))
        net.eval()
        ####
        testdata = get_data(test_data,test_pairs,True)
        test_loader = FastTensorDataLoader(
            *testdata, batch_size=batch_size, shuffle=True)
        
        ####

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[1]) / batch_size))
            test_loss = 0
            preds = []
            alllabel = []
            alltestpairs =[]
            p1 = []
            p2 = []
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels,bathch_indexs in test_loader:
                    bar.update(1)
                    batch_labels =  torch.squeeze(batch_labels)
                    # print('batch_labels=',batch_labels,flush=True)
                    # alllabel.append(batch_labels.item())
                    logits = net(batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = numpy.append(preds, logits.detach().cpu().numpy())
                    alllabel = numpy.append(alllabel, batch_labels.detach().cpu().numpy())
                    
                    ###pairs:
                    x0index,x1index = bathch_indexs.chunk(2,axis=1)
                    p1 = numpy.append(
                                    p1,
                                    x0index.detach().cpu().numpy())
                    p2 = numpy.append(
                                    p2,
                                    x1index.detach().cpu().numpy())
                    x0index = torch.squeeze(x0index).tolist()
                    x1index = torch.squeeze(x1index).tolist()
                    pp = zip(x0index,x1index)
                    for p in pp:
                        alltestpairs.append(p)
                    # for x in x0index:
                    #     p1.append(x)   
                    # for x in x1index:
                    #     p2.append(x)  
                predresult = pd.DataFrame({
                    "p1":p1,
                    "p2":p2,
                    # "pairs": alltestpairs,
                    "labels": alllabel,
                    "preds":preds
                    })
                    # alltestpairs = numpy.append(alltestpairs, bathch_indexs.detach().cpu().numpy())
                    ######
                
                
                test_loss /= test_steps

            ranks,n_prots = compute_rank(predresult)
            Mean_avg_precision = compute_MAP(predresult)
            rk_roc = compute_rank_roc(ranks, n_prots)

            fpr,tpr,thresholds,roc_auc = compute_roc(alllabel, preds)
            accll,topacc,topthre,macc,mthr = acc_thre(thresholds,alllabel,preds,topk=5)
            # precisionll,recallll,thresholdsll = compute_aupr(alllabel, preds)
            
            print('acc list len={}\nmax_acc={}|with_thre={}'.format(len(accll),macc,mthr))
            print('topk acc={}|with_thre={}'.format(topacc,topthre))
            print(f'Test Loss - {test_loss}, AUC - {roc_auc}',flush=True)
            print(f'Test loss - {test_loss}, \nrkAUC - {rk_roc},AUC - {roc_auc}, MAP - {Mean_avg_precision}',flush=True)     

            precisionll,recallll,thresholdsll = compute_aupr(alllabel, preds)
            toppre,toprecall,topthres = aupr_thre(thresholdsll,precisionll,recallll)
            print('toppre={},toprecall={},thres={}'.format(toppre,toprecall,topthres),flush=True)
        

def get_data(df,pairs, newpairsflage = False):
    data = torch.zeros((len(pairs), 2560), dtype=torch.float32)
    labels = torch.zeros((len(pairs), 1), dtype=torch.float32)
    # print('top df=\n',df[:5])
    # print('lendf=',len(df),df[-5:])
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

def topk(k,ll):
    ml = [0]*k
    dl ={}
    for i in range(0,len(ml)):
        dl[ml[i]] = 0
    for i in range(0,len(ml)):
        item=ml[i]
        if item > min(ml):
            dl.pop(min(ml))
            ml.replace(min(ml),item)
            dl[item] = i
            
    return ml

def acc_thre(thresholds,labels, preds,topk=5):
    labels = labels.flatten()
    preds =  preds.flatten()
    acc=[]
    for thre in thresholds:
        predr = numpy.where(preds >= thre,1,0)
        accurancy = sum(predr)/len(predr)
        acc.append(accurancy)
        maxindex = numpy.argmax(numpy.array(acc))
        topkindex = sorted(range(len(acc)), key=lambda i: acc[i])[-topk:]
        topacc = [acc[i] for i in topkindex]
        topthre = [thresholds[i] for i in topkindex] 
    # print('acc list={}\nmax_acc={}|with_thre={}'.format(acc,acc[maxindex],thresholds[maxindex]))
    return acc,topacc,topthre,acc[maxindex],thresholds[maxindex]

def aupr_thre(thresholds,precision,recall,topk=5):
    ## according to precision but not recall
    print('len thre,pre,recall = {}|{}|{}'.format(len(thresholds),len(precision),len(recall)),flush=True)
    topkindex = sorted(range(len(precision)), key=lambda i: precision[i])[-topk:]
    toppre = [precision[i] for i in topkindex]
    toprecall = [recall[i] for i in topkindex]
    topthres = [thresholds[i-1] for i in topkindex] 

    return toppre,toprecall,topthres 



def compute_roc(labels, preds):
    fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,thresholds,roc_auc

def compute_aupr(labels,preds):
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), preds.flatten())
    print('len thre,pre,recall = {}|{}|{}'.format(len(thresholds),len(precision),len(recall)),flush=True)
    
    return precision,recall,thresholds
class Featuredotmodel(nn.Module):
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        # print('dotproduct',dotproduct.size(),flush=True)
        # dotproduct = torch.abs(dotproduct)
        output = self.Sigmoid(dotproduct)
        return output

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
        # dotproduct = torch.abs(dotproduct)

        output = self.Sigmoid(dotproduct)
        return output

if __name__ == '__main__':
  
    main()

