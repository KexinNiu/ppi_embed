import click as ck
import pandas as pd
from torch_utils import FastTensorDataLoader
import torch
import numpy
from torch import nn
from torch.nn import functional as F

# from simdataset import dataset_prosplit
from loaddata import Loaddata,gene_pos,get_posdf  #gene_neg,
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













def main(data_root, batch_size, epochs, hiddim, load, device):
    # model_file = f'{data_root}/deepgo2_esm2.th'
    # out_file = f'{data_root}/predictions_deepgo2_esm2.pkl'
    allinfo_file = f'{data_root}/allinfo.pickle'
    allinter_file = f'{data_root}/allinteractions.pickle'
    loss_func = nn.BCELoss()
    
    # net = Featuremodel(in_dim=1280, hidden_dim=hiddim, out_dim=1280,device=device).to(device)
    net = Featuremodel(in_dim=1280, hidden_dim=hiddim, out_dim=1280)
    print(net,flush=True)
    
    
    if not os.path.exists(allinfo_file):
        print('not exists',flush=True)
        dataset = Loaddata(high_ppipath,low_ppipath,esmfolder,svfolder)
        allinfo = dataset.allinfo
        name1index = allinfo['name']
        interactions = get_posdf(allinfo,dataset.allpairs,dataset.notnegpairs)
        print('sv...',flush=True)
        
        allinfo.to_pickle(allinfo_file)
        interactions.to_pickle(allinter_file)
        print('finished sv',flush=True)

        # indexname = allinfo['name']
    else:

        print('load...',flush=True)
        allinfo = pd.read_pickle(allinfo_file)
        interactions = pd.read_pickle(allinter_file)
        print('>interactions?\n',interactions)
        print('finished loading',flush=True)



    kf = KFold(n_splits = 5, shuffle = True, random_state = 17)
    


    
    for i, (train_index, test_index) in enumerate(kf.split(allinfo)):
        print(f"Fold {i}:")

        train_data, test_data= allinfo.iloc[train_index], allinfo.iloc[test_index]
        print('train top10',train_data[:5])
        train_data = train_data.set_index(train_index) 
        print('train top10',train_data[:5],flush=True)

        print('test_data top10',test_data[:5])
        test_data = test_data.set_index(test_index) 
        print('test_data top10',test_data[:5],flush=True)

        valid_index = train_index[:round(len(train_index)*0.25)]
        valid_data = allinfo.iloc[valid_index]

        print('valid_data top10',valid_data[:5])
        valid_data = valid_data.set_index(valid_index) 
        print('valid_data top10',valid_data[:5],flush=True)


       

        ## gene pos 
       
        # prlist =[set(train_data['name']),set(valid_data['name']),set(test_data['name'])]
        # for i in prlist:
        #     print('len prlist =',len(i))

        # ddposlist = gene_pos(prlist,allpairs)
        # for i in ddposlist:
        #     print('len ddposlist =',len(i.keys()))
        # traindd,valdd,testdd = ddposlist

# TODO: get neg in loaddata

        # ddneglist = gene_neg(prlist,allnotneg,fold=3)
        # ntraindd,nvaldd,ntestdd = ddneglist
        
        def get_tensordata(currdf,inters):
            data =  torch.zeros((len(currdf), 1280), dtype=torch.float32)
            labels = torch.zeros((len(currdf), len(currdf)), dtype=torch.float32)
            labels_name = currdf['name']        
            # print('currdf.info:',currdf.info,flush=True)
            ## name feature
            print('inters:',inters.info)
            for i, row in enumerate(currdf.itertuples()):
                
                data[i, :] = torch.FloatTensor(row.feature)
                print('row name',row.name)
                print('_______')
                print( '?????inters\n', inters)
                print('_______')
                print('inters[row.name]:\n',inters[row.name])# 一列数据
                print('____>>___')

                labels[i] = torch.as_tensor(inters[row.name] )
                print('___!!!____')

                # labels[i,i]= 1
            labels_index = torch.as_tensor(labels)


            # assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
            # print('### data= ',type(data),data)
            # for t in data:
            #     print('t.shape[0]=',t.shape[0])
            #     print('tensors[0].shape[0]',data[0].shape[0])
            #     break

            # # labels_index = torch.as_tensor(labels_index)
            # # labels_index = labels_index.unsqueeze(0).t()

            # print('### labels_index= ',type(labels_index),labels_index)
            # for t in labels_index:
            #     # if t.shape[0] != labels_index[0].shape[0]:
            #     #     print('errp?',labels_index[0].shape[0])
            #     print('t.shape[0]=',t.shape[0],t.shape[0])
            #     print('tensors[0].shape[0]',labels_index[0].shape[0])
                
            # print ('++++++++++++++++++++++++++++++')

            # print('typedata= ',type(data),data.size)
            # print('type  labels= ',type(labels_index),labels_index.shape)
            # print('type  labels= ', labels_index[0].shape,labels_index[0].shape)
            return data,labels_index,labels_name
            # return data,inters,labels_name


        def get_posddinfo(posdd,train_labelindex,train_labelname):
            for i in range (0,len(train_labelname)):
                prname = train_labelname.loc[i]
                print('prname',prname)
                # prindex = i
                print('posdd',posdd)
                pospr = posdd[prname]
                print('pospr',pospr)

                print('train)_labelname=',train_labelname)
                posprindex = [train_labelname[train_labelname.name ==x] for x in pospr]
                print('posprindex',posprindex)
                
                train_labelname.loc[i][posprindex] =  1

            return train_labelindex


        train_data,train_labelindex,train_labelname = get_tensordata(train_data,interactions)
        # train_labelindex = get_posddinfo(traindd,train_labelindex,train_labelname)
        ##add pos protein  info to label index

        
        traindata = train_data, train_labelindex
        
        train_loader = FastTensorDataLoader(
            *traindata, batch_size=batch_size, shuffle=True)

        for batch_features, batch_labels in train_loader:
            print('batch_features=',batch_features,batch_features.size())
            print('batch_labels=',batch_labels,batch_labels.size())
            break
        print('+++++++++++++++++++++++++++=====',flush=True)
        a = 1
        b= 2
        assert a == b

# ---------------------------------------------------------------------

        # valid_loader = FastTensorDataLoader(
        #     *valid_data, batch_size=batch_size, shuffle=False)
        # test_loader = FastTensorDataLoader(
        #     *test_data, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
        scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)

        best_loss = 100000.0
        if not load:
            print('Training the model from the begnning',flush=True)
            for epoch in range(epochs):
                net.train()
                train_loss = 0
                lmbda = 0.1
                train_steps = int(math.ceil(len(train_data) / batch_size))
                with ck.progressbar(length=train_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in train_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)

                        logits = net(batch_features)
                        loss = F.binary_cross_entropy(logits, batch_labels)

                        total_loss = loss
                        train_loss += loss.detach().item()
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        
                train_loss /= train_steps
                
                print('Validation')
                net.eval()
                with torch.no_grad():
                    valid_steps = int(math.ceil(len(valid_data) / batch_size))
                    valid_loss = 0
                    preds = []
                    with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                        for batch_features, batch_labels in valid_loader:
                            bar.update(1)
                            batch_features = batch_features.to(device)
                            batch_labels = batch_labels.to(device)
                            logits = net(batch_features)
                            batch_loss = F.binary_cross_entropy(logits, batch_labels)
                            valid_loss += batch_loss.detach().item()
                            preds = np.append(preds, logits.detach().cpu().numpy())
                    valid_loss /= valid_steps
                    roc_auc = compute_roc(valid_labels, preds)
                    print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model')
                    th.save(net.state_dict(), model_file)

 
        
        
        break
        

    
    





    # terms_dict, train_data, valid_data, test_data, test_df = load_data(data_root, ont, terms_file)
    # n_terms = len(terms_dict)
    # train_features, train_labels = train_data
    # valid_features, valid_labels = valid_data
    # test_features, test_labels = test_data
    
    # train_loader = FastTensorDataLoader(
    #     *train_data, batch_size=batch_size, shuffle=True)
    # valid_loader = FastTensorDataLoader(
    #     *valid_data, batch_size=batch_size, shuffle=False)
    # test_loader = FastTensorDataLoader(
    #     *test_data, batch_size=batch_size, shuffle=False)
    
    # valid_labels = valid_labels.detach().cpu().numpy()
    # test_labels = test_labels.detach().cpu().numpy()
    
    # optimizer = th.optim.Adam(net.parameters(), lr=3e-4)
    # scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)

    # best_loss = 10000.0
    # if not load:
    #     print('Training the model')
    #     for epoch in range(epochs):
    #         net.train()
    #         train_loss = 0
    #         lmbda = 0.1
    #         train_steps = int(math.ceil(len(train_labels) / batch_size))
    #         with ck.progressbar(length=train_steps, show_pos=True) as bar:
    #             for batch_features, batch_labels in train_loader:
    #                 bar.update(1)
    #                 batch_features = batch_features.to(device)
    #                 batch_labels = batch_labels.to(device)
    #                 logits = net(batch_features)
    #                 loss = F.binary_cross_entropy(logits, batch_labels)
    #                 total_loss = loss
    #                 train_loss += loss.detach().item()
    #                 optimizer.zero_grad()
    #                 total_loss.backward()
    #                 optimizer.step()
                    
    #         train_loss /= train_steps
    #         print('Validation')
    #         net.eval()
    #         with th.no_grad():
    #             valid_steps = int(math.ceil(len(valid_labels) / batch_size))
    #             valid_loss = 0
    #             preds = []
    #             with ck.progressbar(length=valid_steps, show_pos=True) as bar:
    #                 for batch_features, batch_labels in valid_loader:
    #                     bar.update(1)
    #                     batch_features = batch_features.to(device)
    #                     batch_labels = batch_labels.to(device)
    #                     logits = net(batch_features)
    #                     batch_loss = F.binary_cross_entropy(logits, batch_labels)
    #                     valid_loss += batch_loss.detach().item()
    #                     preds = np.append(preds, logits.detach().cpu().numpy())
    #             valid_loss /= valid_steps
    #             roc_auc = compute_roc(valid_labels, preds)
    #             print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}')

    #         if valid_loss < best_loss:
    #             best_loss = valid_loss
    #             print('Saving model')
    #             th.save(net.state_dict(), model_file)

    #         #scheduler.step()
            

    # # Loading best model
    # print('Loading the best model')
    # net.load_state_dict(th.load(model_file))
    # net.eval()
    # with th.no_grad():
    #     test_steps = int(math.ceil(len(test_labels) / batch_size))
    #     test_loss = 0
    #     preds = []
    #     with ck.progressbar(length=test_steps, show_pos=True) as bar:
    #         for batch_features, batch_labels in test_loader:
    #             bar.update(1)
    #             batch_features = batch_features.to(device)
    #             batch_labels = batch_labels.to(device)
    #             logits = net(batch_features)
    #             batch_loss = F.binary_cross_entropy(logits, batch_labels)
    #             test_loss += batch_loss.detach().cpu().item()
    #             preds = np.append(preds, logits.detach().cpu().numpy())
    #         test_loss /= test_steps
    #     preds = preds.reshape(-1, n_terms)
    #     roc_auc = compute_roc(test_labels, preds)
    #     print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

        
    # preds = list(preds)
    # # Propagate scores using ontology structure
    # for i, scores in enumerate(preds):
    #     prop_annots = {}
    #     for go_id, j in terms_dict.items():
    #         score = scores[j]
    #         for sup_go in go.get_anchestors(go_id):
    #             if sup_go in prop_annots:
    #                 prop_annots[sup_go] = max(prop_annots[sup_go], score)
    #             else:
    #                 prop_annots[sup_go] = score
    #     for go_id, score in prop_annots.items():
    #         if go_id in terms_dict:
    #             scores[terms_dict[go_id]] = score

    # test_df['preds'] = preds

    # test_df.to_pickle(out_file)






class Featuremodel(nn.Module):
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
        x = self.linear1(x)
        x = self.Sigmoid(x)
        x = self.linear2(x)
        # x = self.feature_norm(x)
        return x


if __name__ =='__main__':
    
    high_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.highconf.links.txt'
    low_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.lowconf.links.txt'
    esmfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/'
    svfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/'

    main()
