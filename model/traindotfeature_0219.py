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
from sklearn.metrics import roc_curve, auc, matthews_corrcoef

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
    '--learningrate', '-lr',default = 1e-4,
    help='learning rate') 


# data_root ="/home/niuk0a/Documents/COOOODE/simembed/new/"  
# high_ppipath='/home/niuk0a/Documents/COOOODE/simembed/4932.highconf.links.txt'
# low_ppipath='/home/niuk0a/Documents/COOOODE/simembed/4932.lowconf.links.txt'
# esmfolder='xxxxxxrandom gene'
# svfolder='/home/niuk0a/Documents/COOOODE/simembed/new/'


def main(data_root, batch_size, epochs, hiddim, load, device, negativefold,name,esmfolder, modelname,learningrate):
    print('>>>',torch.cuda.is_available())
    lr= learningrate
    print('lr=',lr,flush=True)
    print('batch_size=',batch_size,flush=True)
    model_file = f'{data_root}/dotf_esm2{modelname}_{batch_size}.th'
    out_file = f'{data_root}/predictions_{modelname}{batch_size}dotproductsim_esm2.pkl'
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
        print('all pairs count number ={}'.format(len))
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
    if modelname == 'onef':
        net = Featuredotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280) 
    elif modelname =='twof':
        net = twoFeaturedotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280) 
    elif modelname =='twofupdate':
        net = twoupFeaturedotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280) 
    loss_func = nn.BCELoss()
    # lr = 1e-4
    print('lr=',lr,flush=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)
    print('>>>model setting done\n',flush=True)

    
    # model 
    ########################################

    kf = KFold(n_splits = 5, shuffle = True, random_state = 17)
    for i, (train_index, test_index) in enumerate(kf.split(allinfo)):
        print(f"Fold {i}:",flush=True)
        valid_index = train_index[:round(len(train_index)*0.25)]
        print('train index={}'.format(train_index))
        train_index1 = list(set(train_index).difference(set(train_index[:round(len(train_index)*0.25)])))
       
        train_data= allinfo.iloc[train_index1]
        test_data= allinfo.iloc[test_index]
        valid_data = allinfo.iloc[valid_index]

        train_pairs = allpairs[(allpairs['p1'].isin(train_index1)) & (allpairs['p2'].isin(train_index1))]
        test_pairs = allpairs[(allpairs['p1'].isin(test_index)) & (allpairs['p2'].isin(test_index))]
        valid_pairs = allpairs[(allpairs['p1'].isin(valid_index)) & (allpairs['p2'].isin(valid_index))]
        # print('>>>top10 train \n',train_pairs[:5],train_data[:5],train_data[-5:])
        # print('>>>top10 test \n',test_pairs[:5],test_data[:5],test_data[-5:])
        
        # train_data = train_data.set_index(train_index) 
        # test_data = test_data.set_index(test_index) 
        # valid_data = valid_data.set_index(valid_index) 
        # print('after set index\n\n')
        # print('>>>top10 train \n',train_pairs[:5],train_data[:5],train_index1[:5])
        # print('>>>top10 test \n',test_pairs[:5],test_data[:5],test_index[:5])

        traininfo_file = f'{data_root}/{name}_{i}_traininfo.pkl'
        trainpairs_file = f'{data_root}/{name}_{i}_trainpairs.pkl'
        # name2index_file = f'{data_root}/{name}_name2index.pkl'
        # index2name_file = f'{data_root}/{name}_index2name.pkl'

        train_data.to_pickle(traininfo_file)
        train_pairs.to_pickle(trainpairs_file)

        traindata = get_data(train_data,train_pairs)
        train_loader = FastTensorDataLoader(
            *traindata, batch_size=batch_size, shuffle=True)
        
        validdata = get_data(valid_data,valid_pairs)

        valid_loader = FastTensorDataLoader(
            *validdata, batch_size=batch_size, shuffle=True)
        # data prepare##
        print(' # data prepare##',flush=True)
        #####################################################
        print('#####################################################')
        print('test df={}'.format(len(test_data.index)))
        print('vilad df={}'.format(len(valid_data.index)))
        print('train df={}'.format(len(train_data.index)))
        print('#####################################################')

        
        best_loss = 100000.0
        if not load:
            net.train()
            currdf = train_data
            print('Training the model from the begnning',flush=True)
            for epoch in range(epochs):
                print('train {}th epochs'.format(epoch),flush=True)
                train_loss = 0
                pt =0
                train_steps = int(math.ceil(len(train_data) / batch_size))
                print('train _data')
                print('train {}currdf={}'.format(epoch,len(currdf.index)))
                # print('traincurrdf=\n{}'.format(currdf[:5]))
                # if epoch ==0:
                #     tmpcurrdf = currdf.index
                # if epoch ==1:
                #     newcurrdf = currdf.index
                #     print('ori len={},and new len of inde ={}'.format(len(tmpcurrdf),len(newcurrdf)))


                with ck.progressbar(length=train_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in train_loader:
                        bar.update(1)
                        pt+=1
                        # batch_features = batch_features.to(device)
                        # batch_labels = batch_labels.to(device)
                        # batch_features = batch_features
                        # batch_labels = batch_labels
                        batch_labels =  torch.squeeze(batch_labels)

                        ###!!!####
                        # print('batch_labels\t',batch_labels,batch_labels.size(),flush=True)
                        # print('batch_features\t',batch_features,batch_features.size(),flush=True)
                        ###!!!####

                        # fflg, batch_features,batch_labels = check_batch(batch_features,batch_labels,currdf) 
                        # if not fflg:
                        #     # print('empty batch')
                        #     continue
                        
                        
                        
                        # output = net(batch_features,currdf,upd=True)
                        # print('train df={}'.format(len(currdf.index)))

                        output = net(batch_features)

                        # print('output\t',output,output.size(),flush=True)
                        try:
                            loss = F.binary_cross_entropy(output, batch_labels)
                        except:
                            batch_labels = torch.unsqueeze(batch_labels, 0)
                            # print('__________________\n',flush=True)
                            # print('batch_labels=\n',batch_labels,batch_labels.size())
                            # print('batch_features=\n',batch_features)
                            # print('output=\n',output,output.size())
                            # print('__________________\n',flush=True)
                            loss = F.binary_cross_entropy(output, batch_labels)
                            # print('__________________\n',flush=True)



                        total_loss = loss
                        train_loss += loss.detach().item()
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        if pt % 500 ==0:
                            print('batch loss=',total_loss.item(),flush=True)
                train_loss /= train_steps
                print('>>Validation',flush=True)
                net.eval()
                with torch.no_grad():
                    valid_steps = int(math.ceil(len(valid_data) / batch_size))
                    valid_loss = 0
                    preds = []
                    ##
                    validcurrdf = valid_data
                    ##
                    with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                        for batch_features, batch_labels in valid_loader:
                            bar.update(1)
                            # batch_features = batch_features.to(device)
                            # batch_labels = batch_labels.to(device)
                            # batch_features = batch_features
                            # batch_labels = batch_labels
                            batch_labels =  torch.squeeze(batch_labels)
                            logits = net(batch_features)
                            # fflg, batch_features,batch_labels = check_batch(batch_features,batch_labels,validcurrdf) 
                            # if not fflg:
                            #     print('empty batch')
                            #     continue
                            # logits = net(batch_features,validcurrdf)
                            # print('valid df={}'.format(len(validcurrdf.index)))


                            batch_loss = F.binary_cross_entropy(logits, batch_labels)
                            valid_loss += batch_loss.detach().item()
                            preds = numpy.append(preds, logits.detach().cpu().numpy())
                    valid_loss /= valid_steps
                    roc_auc = compute_roc(validdata[1], preds)
                    print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}',flush=True)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model',flush=True)
                    torch.save(net.state_dict(), model_file)

                scheduler.step()

        # Loading best model
        print('..Loading the best model')
        net.load_state_dict(torch.load(model_file))
        net.eval()
        ####
        testdata = get_data(test_data,test_pairs)
        test_loader = FastTensorDataLoader(
            *testdata, batch_size=batch_size, shuffle=True)
        
        ####

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[1]) / batch_size))
            test_loss = 0
            preds = []
            testcurrdf = test_data
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels in test_loader:
                    bar.update(1)
                    # batch_features = batch_features.to(device)
                    # batch_labels = batch_labels.to(device)
                    batch_labels =  torch.squeeze(batch_labels)
                    
                    logits = net(batch_features)
                    # fflg, batch_features,batch_labels = check_batch(batch_features,batch_labels,testcurrdf) 
                    # if not fflg:
                    #     continue
                        # print('empty batch')
                        # continue
                    # logits  = net(batch_features,testcurrdf)
                    # print('test df={}'.format(len(testcurrdf.index)))
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = numpy.append(preds, logits.detach().cpu().numpy())
                test_loss /= test_steps
            # preds = preds.reshape(-1, n_terms)?????? 
            roc_auc = compute_roc(testdata[1], preds)
            print(f'Test Loss - {test_loss}, AUC - {roc_auc}',flush=True)
        test_pairs['preds'] = preds
        test_pairs.to_pickle(out_file)


def get_data(df,pairs):
    # data = torch.zeros((len(pairs), 2560), dtype=torch.float32)
    data = torch.zeros((len(pairs), 2), dtype=torch.float32)
    labels = torch.zeros((len(pairs), 1), dtype=torch.float32)
    # print('top df=\n',df[:5])
    # print('lendf=',len(df),df[-5:])

    for i, row in enumerate(pairs.itertuples()):

        
        f1 = row.p1
        f2 = row.p2
        try:
            fff = df.loc[df['index']==f1]
            fff = df.loc[df['index']==f2]
        except:
            continue
        f1 = torch.tensor(f1,dtype=torch.float32)
        f2 = torch.tensor(f2,dtype=torch.float32)
        
        # try:
        #     f1 = df.loc[df['index']==row.p1].feature.values.tolist()[0]
        #     f2 = df.loc[df['index']==row.p2].feature.values.tolist()[0]
        #     f1 = torch.tensor(f1)
        #     f2 = torch.tensor(f2)
        # except:
        #     # print('index p1 =',row.p1)
        #     # print('index p2 =',row.p2,flush=True)
        #     continue
        # print('f1={},f2 ={}'.format(f1.size(),f2.size()),flush=True)
        f1 = f1.unsqueeze(axis=0)
        f2 = f2.unsqueeze(axis=0)
        # print('f1={},f2 ={}'.format(f1.size(),f2.size()),flush=True)

        newf = torch.cat((f1,f2),0)
        # print('newf=',newf)
        # ff1,ff2 = newf.chunk(2)
        data[i,:] = torch.FloatTensor(newf)
        labels[i,:] = torch.tensor(row.label)
        
    return data,labels

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

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
        output = self.Sigmoid(dotproduct)
        return output

class twoFeaturedotmodel(nn.Module):
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
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)

        x1 = self.linear3(x1)
        x1 = self.Sigmoid(x1)
        x1 = self.linear4(x1)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        # print('dotproduct',dotproduct.size(),flush=True)
        output = self.Sigmoid(dotproduct)
        return output,x0,x1

class twoupFeaturedotmodel(nn.Module):
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
        # self.feature_norm = nn.LayerNorm()
    def forward(self,x:torch.Tensor,df,upd=False)-> torch.Tensor:
        x0index,x1index = x.chunk(2,axis=1)
        # print('x0index=',x0index[:3])
        
        x0index = torch.squeeze(x0index).tolist()
        x1index = torch.squeeze(x1index).tolist()
        if isinstance(x0index, float):
            ff0 = df.loc[df['index']==x0index]
            ff1 = df.loc[df['index']==x1index]
            x0 = torch.tensor(ff0['feature'].values[0])
            x1 = torch.tensor(ff1['feature'].values[0])
        else:
            x0c,x1c = [],[]
            for i in range(0,len(x0index)):
                ff0 = df.loc[df['index']==x0index[i]]
                ff1 = df.loc[df['index']==x1index[i]]
                x0c.append(torch.tensor(ff0['feature'].values[0]))
                x1c.append(torch.tensor(ff1['feature'].values[0]))
                x0 = torch.stack(x0c)
                x1 = torch.stack(x1c)

        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)

        x1 = self.linear3(x1)
        x1 = self.Sigmoid(x1)
        x1 = self.linear4(x1)

        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        # print('dotproduct',dotproduct.size(),flush=True)
        output = self.Sigmoid(dotproduct)
        if upd ==True:
            if isinstance(x0index, float):
                tmp0 = x0.detach().numpy().tolist()
                tmp1 = x1.detach().numpy().tolist()
                mask = df['index'] == x0index
                mask1 = df['index'] == x1index

                df.loc[mask,'feature'] = pd.Series([tmp0], index=df.index[mask])
                df.loc[mask1,'feature'] = pd.Series([tmp1], index=df.index[mask1])
            else:
                # mask = df['index'] == x0index
                # mask1 = df['index'] == x1index
                # df.loc[mask,'feature'] = pd.Series([x0], index=df.index[mask])
                # df.loc[mask1,'feature'] = pd.Series([x1], index=df.index[mask1])
                 
                for i in range(0,len(x0index)):
                    tmp0 = x0[i].detach().numpy().tolist()
                    tmp1 = x1[i].detach().numpy().tolist()
                    
                    mask = df['index'] == x0index[i]
                    mask1 = df['index'] == x1index[i]

                    df.loc[mask,'feature'] = pd.Series([tmp0], index=df.index[mask])
                    df.loc[mask1,'feature'] = pd.Series([tmp1], index=df.index[mask1])
                    # print('assgin finished!')



        return  output


def check_batch(batch_features,batch_labels,df):
    # print('ori batch labels={}\n'.format(batch_labels.size()))
    # print('ori batch feature size=\n{}'.format(batch_features.size()))
    x0index,x1index = batch_features.chunk(2,axis=1)
    # print('x0index=',x0index[:3])
    x0index = torch.squeeze(x0index).tolist()
    x1index = torch.squeeze(x1index).tolist()
    indexlist = df['index'].tolist()
    # print('indexlist={}|q{}'.format(indexlist[:5],len(indexlist)))
    if isinstance(x0index, float):
        if x0index in indexlist and x1index in indexlist:
            # print('new batch labels={}\n'.format(batch_labels.size()))
            # print('new batch feature size=\n{}'.format(batch_features.size()))

            return True,batch_features,batch_labels
        else:
            return False,1,1
    else:
        newx0,newx1,newlabel = [],[],[]
        for i in range(0,len(x0index)):
            if x0index[i] in indexlist and x1index[i] in indexlist:
                newx0.append(x0index[i])
                newx1.append(x1index[i])
                newlabel.append(batch_labels[i])
        if newlabel:
            newx0 = torch.tensor(newx0,dtype=torch.float32)
            newx1 = torch.tensor(newx1,dtype=torch.float32)
            newf = torch.stack((newx0,newx1),1)
            newlabel = torch.stack(newlabel)
            # print('new batch labels={}\n'.format(newlabel.size()))
            # print('new batch feature size=\n{}'.format(newf.size()))
            return True,newf,newlabel
        else:
            return False,1,1



# f1 = torch.tensor(f1,dtype=torch.float32)
# f2 = torch.tensor(f2,dtype=torch.float32)              
# f1 = f1.unsqueeze(axis=0)
# f2 = f2.unsqueeze(axis=0)
# newf = torch.cat((f1,f2),0)
            



    


if __name__ == '__main__':

    main()

