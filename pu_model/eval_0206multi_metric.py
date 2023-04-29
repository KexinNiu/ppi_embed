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

from network import Featuredotmodel,twoFeaturedotmodel,PUloss
from utils import topk,acc_thre,aupr_thre,compute_aupr,compute_roc
from utils_data import get_data,get_test_data
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
    







def main(data_root, batch_size, epochs, hiddim, load, device, negativefold,name,esmfolder, modelname,metric):
    print('>>>',torch.cuda.is_available())
    print('batch_size=',batch_size,flush=True)

    # model_file = f'{data_root}/model_esm2_{epochs}_{modelname}_{metric}_{batch_size}.th'
    model_file = f'{data_root}/model_esm2_rk0408_{epochs}_{modelname}_{metric}_{batch_size}.th'
    out_file = f'{data_root}/predictions_rk0408_{epochs}_{modelname}_{metric}_{batch_size}.pkl'
    eval_file = f'{data_root}/predictions_E@10_rk0408_{epochs}_{modelname}_{metric}_{batch_size}.txt'

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
        
    loss_func = PUloss
    net.eval()

    print('>>>model setting done\n',flush=True)

    def filter_protein_contain_posPPI(allinfo,allpairs):
        posinteractions = allpairs.query('labels==1.0')
        print('posinteractions={}'.format(posinteractions[:3]),flush=True)
        
        pos_proteins = numpy.unique(posinteractions[["protein1", "protein2"]].values)
        print('pos_proteins={}'.format(pos_proteins),flush=True)

        posinfo = allinfo[pos_proteins]
        print('posinfo={}'.format(posinfo[:3]),flush=True)
        remaininfo = allinfo[~pos_proteins]
        print('remaininfo={}'.format(remaininfo[:3]))


        return posinfo, remaininfo
    
    print('allinfo={}'.format(allinfo[:3]))
    
    posinfo,remaininfo = filter_protein_contain_posPPI(allinfo,allpairs)
 

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
        # testdata = get_data(test_data,test_pairs,True)
        print('original test len={}'.format(len(test_index)),flush=True)
        # test_data= allinfo.iloc[test_index[:800]]
        # test_pairs = allpairs[(allpairs['p1'].isin(test_index[:800])) & (allpairs['p2'].isin(test_index[:800]))]
        testdata = get_test_data(test_data,test_pairs,True)

        test_loader = FastTensorDataLoader(
            *testdata, batch_size=batch_size, shuffle=True)
        
        ####

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[1]) / batch_size))
            test_loss = 0
            preds = []
            alllabel = []
            alltestpairs =[]
            
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels,bathch_indexs in test_loader:
                    bar.update(1)
                    # print('batch_labels=',batch_labels,flush=True)

                    batch_labels =  torch.squeeze(batch_labels)
                    # print('batch_labels=sq=',batch_labels,flush=True)
                    # alllabel.append(batch_labels.item())
                    logits = net(batch_features)
                    batch_loss = loss_func(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = numpy.append(preds, logits.detach().cpu().numpy())
                    alllabel = numpy.append(alllabel, batch_labels.detach().cpu().numpy())
                    
                    ###pairs:
                    x0index,x1index = bathch_indexs.chunk(2,axis=1)
                    x0index = torch.squeeze(x0index).tolist()
                    x1index = torch.squeeze(x1index).tolist()
                    pp = zip(x0index,x1index)
                    for p in pp:
                        alltestpairs.append(p)
                    # alltestpairs = numpy.append(alltestpairs, bathch_indexs.detach().cpu().numpy())
                    ######
                print('len current len of all testpairs=',len(alltestpairs),flush=True)
                test_loss /= test_steps

            predresult = pd.DataFrame({
            "pairs": alltestpairs,
            "labels": alllabel,
            "preds":preds
            })
            predresult.to_pickle(out_file)

            print('>>>>alllabel=',alllabel,len(alllabel),flush=True)
            print('>>>>preds=',preds,len(preds),flush=True)
            preds1 = preds[numpy.where(alllabel!=2.0)]
            print('preds1=',preds1,flush=True)
            def sigmoid(x):
                return 1/(1+numpy.exp(-x))
            preds1 = sigmoid(preds1)
            print('preds1=',preds1,flush=True)
            alllabel1 = alllabel[numpy.where(alllabel!=2.0)]
            print('alllabel1=',alllabel1,'sum=',sum(alllabel1),flush=True)

            _,_,_,roc_auc = compute_roc(alllabel1, preds1)

            print(f'Test Loss - {test_loss}, AUC - {roc_auc}',flush=True)

            # predresult = pd.DataFrame({
            # "pairs": alltestpairs,
            # "labels": alllabel,
            # "preds":preds
            # })
            # predresult.to_pickle(out_file)
            eval_ranking(out_file,eval_file)

def eval_ranking(predp,outp):
    outf = open(outp,'w')
    # p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
    predresult = pd.read_pickle(predp)
    predresult['protein1'] = predresult['pairs'].map(lambda x:int(x[0]))
    predresult['protein2'] = predresult['pairs'].map(lambda x:int(x[1]))
    aa = predresult.sort_values(by=['protein1','protein2'])
    # print('>>>>>>\n',aa[:5])
    unique_proteins = numpy.unique(aa[["protein1", "protein2"]].values)
    a1_list=[]
    a10_list=[]
    n=0
    print('total={}'.format(len(unique_proteins)))
    for protein in unique_proteins:
        n+=1
        if n % (len(unique_proteins)//10)==0:
            print('>>{}/{}'.format(n,len(unique_proteins)))
        allinters = aa.query("protein1==@protein | protein2==@protein")
        allinters = allinters.sort_values(by=['preds'],ascending=False)
        # negvals = allinters.query('labels==0.0')['preds']
        posvals = allinters.query('labels==1.0')['preds']
        # print('posvals=\n',posvals)
        if len(posvals) >=1:
            # print('>>>>>all top10\n',allinters.sort_values(by=['preds'],ascending=False)[:10])
            at1 = allinters['labels'].values[0]
            
            if len(posvals) >=10:
                top10 = allinters[:10]
                at10 = sum(top10['labels'])
                a1_list.append(at1)
                a10_list.append(at10)
                print('{} \t @1={} @10={}'.format(protein,at1,at10),file=outf)
            else:
                print('{} \t @1={} @10=None'.format(protein,at1),file=outf,flush=True)

    print('a1_list\ntotal={},hit@1={}'.format(len(a1_list),sum(a1_list)/len(a1_list)),file=outf)
    print('a10_list\ntotal={},hit@10={}'.format(len(a10_list),sum(a10_list)/(10*len(a10_list))),file=outf)

if __name__ == '__main__':
  
    main()

