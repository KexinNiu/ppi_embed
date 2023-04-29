import math
import datetime
import torch
import click as ck
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import MultiStepLR
from datapreprocess import extract_esm
from network import Featuredotmodel, twoFeaturedotmodel,PUloss,tri_PUloss
from utils_data import get_traindata,get_val_testdata,get_index_val_testdata,get_index_traindata
from torch_utils import FastTensorDataLoader
from utils import compute_rank,compute_rank_roc,compute_roc,compute_MAP

import os
@ck.command()

@ck.option(
    '--modelname', '-mn',
    help='name of model,one,two')
@ck.option(
    '--data-root', '-dr', default='data',
    help='ppi and interactions folder root')
@ck.option(
    '--rawdata_flage', '-fl', is_flag=True,
    help='from sequences data/STRING RAW DATA')
@ck.option(
    '--fasta-fp', '-fp', 
    help='fasta_fp')
@ck.option(
    '--ppi-fp', '-pp',  
    help='ppi_fp')
@ck.option(
    '--hiddim', '-hd', default=128,
    help='hidden dimension of the model') 
@ck.option(
    '--indim', '-ind', default=2560,
    help='input dimension of the model') 
@ck.option(
    '--outdim', '-outd', default=2560,
    help='output dimension of the model') 
@ck.option(
    '--batch-size', '-bs', default=16,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--learningrate', '-lr', default=1e-4,
    help='learning rate') 
# @ck.option(
#     '--device', '-dv', default='cuda:0',
#     help='Device') 


def main(modelname,data_root,rawdata_flage,fasta_fp,ppi_fp,indim, hiddim,outdim,learningrate,epochs,batch_size):
    
    # ESM features preparation...
    print('\nESM features preparation...',datetime.datetime.now(),flush=True)

    proteins_fp = f'{data_root}/esm/proteins_name2index.pkl'
    esm_fp = f'{data_root}/esm/esm_proteins.pkl'

    if rawdata_flage==True:
        if not os.path.exists(proteins_fp) and not os.path.exists(esm_fp):
            # Extract ESM features
            esm_dir = Path(f'{data_root}/esm')
            proteins, data = extract_esm(fasta_fp, output_dir=esm_dir)
            # 这个extraction有问题 极慢 但是有可能是因为用了cpu不是gpu
            print('ESM features extracted...',datetime.datetime.now(),flush=True)

            proteins_name2index = {proteins[i]: i for i in range(len(proteins))}
            pd.to_pickle(proteins_name2index,proteins_fp)

            proteinsname_fp = proteins_fp.replace('name2index','namelist')
            pd.to_pickle(proteins,proteinsname_fp)
            pd.to_pickle(data,esm_fp)
            print('ESM features saved...',datetime.datetime.now(),flush=True)

        else:
            proteinsname_fp = proteins_fp.replace('name2index','namelist')
            print('ESM features loading...',datetime.datetime.now(),flush=True)
            proteins =  pd.read_pickle(proteinsname_fp)
            proteins_name2index = pd.read_pickle(proteins_fp)
            data = pd.read_pickle(esm_fp)
            print('ESM features loaded...',datetime.datetime.now(),flush=True)


    else:
        proteinsname_fp = proteins_fp.replace('name2index','namelist')
        print('ESM features loading...',datetime.datetime.now(),flush=True)
        proteins =  pd.read_pickle(proteinsname_fp)
        proteins_name2index = pd.read_pickle(proteins_fp)
        data = pd.read_pickle(esm_fp)
        print('ESM features loaded...',datetime.datetime.now(),flush=True)

    
    # STRING interactions preparation
    interactions_index_fp = f'{data_root}/esm/interactions.pkl'
    notneg_interactions_index_fp = f'{data_root}/esm/interactions_notneg.pkl'
    posproteins_index_fp = f'{data_root}/esm/pos_proteins_index.pkl'

    if rawdata_flage==True:
        # Read from raw STRING .txt file
        print('STRING interactions preparation...',datetime.datetime.now(),flush=True)
        if not os.path.exists(interactions_index_fp) and not os.path.exists(posproteins_index_fp):
            interactions = pd.read_csv(ppi_fp, delim_whitespace=True)

            # Match the Proteins with index
            interactions['protein1'] = interactions['protein1'].map(lambda x:int(proteins_name2index[x]))
            interactions['protein2'] = interactions['protein2'].map(lambda x:int(proteins_name2index[x]))

            # Filter for high confidence interactions
            high_confidence_threshold = 700
            interactions = interactions[interactions["combined_score"] >= high_confidence_threshold]
            notneg_interactions = interactions[interactions["combined_score"] < high_confidence_threshold]
            print('STRING interactions filtered...',datetime.datetime.now(),flush=True)
            
            # Get a list of unique proteins which have positive interactions
            unique_proteins = np.unique(interactions[["protein1", "protein2"]].values)

            pd.to_pickle(interactions,interactions_index_fp)
            pd.to_pickle(notneg_interactions,notneg_interactions_index_fp)
            pd.to_pickle(unique_proteins,posproteins_index_fp)
            print('STRING interactions saved...',datetime.datetime.now(),flush=True)
        else:
            print('STRING interactions loading...',datetime.datetime.now(),flush=True)
            interactions = pd.read_pickle(interactions_index_fp)
            notneg_interactions = pd.read_pickle(notneg_interactions_index_fp)
            unique_proteins = pd.read_pickle(posproteins_index_fp)
            print('STRING interactions loaded...',datetime.datetime.now(),flush=True)



    else:
        print('STRING interactions loading...',datetime.datetime.now(),flush=True)

        interactions = pd.read_pickle(interactions_index_fp)
        notneg_interactions = pd.read_pickle(notneg_interactions_index_fp)
        unique_proteins = pd.read_pickle(posproteins_index_fp)
        print('STRING interactions loaded...',datetime.datetime.now(),flush=True)


    proteins_index = [i for i in range(len(proteins))] ## all proteins
    rproteins_index = [i for i in proteins_index if i not in unique_proteins] ## proteins no positive interactions

    ######################################
    ####### train/val/test splite ########
    ######################################

    # specify the number of folds
    K = 5

    # split data1 and data2 into K-folds
    kf = KFold(n_splits=K, shuffle=True, random_state=17)

    data1_splits = kf.split(unique_proteins)
    data2_splits = kf.split(rproteins_index)
    print('\n\n\n>>>>>>>>Train...',datetime.datetime.now(),flush=True)

    for fold in range(K):
        if fold == 1:
            break
        print('Fold={}'.format(fold),datetime.datetime.now(),flush=True)
        train_index1, test_index1 = next(data1_splits)
        train_index2, test_index2 = next(data2_splits)

        train_prot_index = [unique_proteins[x] for x in train_index1] \
                            + [rproteins_index[x] for x in train_index2]
        test_prot_index = [unique_proteins[x] for x in test_index1]  \
                            + [rproteins_index[x] for x in test_index2]
        print('test_prot_index={}=={}'.format(len(test_prot_index),len(np.unique(test_prot_index))),flush=True)
        print('train_prot_index={}=={}'.format(len(train_prot_index),len(np.unique(train_prot_index))),flush=True)
        
        # train_esm_data = [data[i] for i in train_prot_index]
        # test_esm_data  = [data[i] for i in test_prot_index]
        train_esm_data={}
        test_esm_data={}
        for i in train_prot_index:
            train_esm_data[i] = data[i]
        for i in test_prot_index:
            test_esm_data[i] = data[i]
        # train_esm_data = {train_esm_data[i]=data[i] for i in train_prot_index}
        # test_esm_data  = {i:data[i] for i in test_prot_index}

        val_proteins, train_proteins = \
                        train_prot_index[:round(len(train_prot_index) * 0.25)],\
                        train_prot_index[round(len(train_prot_index) * 0.25)+1:]
        test_proteins = test_prot_index

        train_pairs = interactions[(interactions['protein1'].isin(train_proteins))
                               & (interactions['protein2'].isin(train_proteins))]
        valid_pairs = interactions[(interactions['protein1'].isin(val_proteins))
                               & (interactions['protein2'].isin(val_proteins))]
        test_pairs = interactions[(interactions['protein1'].isin(test_proteins))
                               & (interactions['protein2'].isin(test_proteins))]
        print('data prpeparation down...',datetime.datetime.now(),flush=True)



        #########

        # print('>>>Validation',datetime.datetime.now(), flush=True)
        # ### train loader
        # valdata = get_index_val_testdata(val_proteins, train_esm_data,valid_pairs,notneg_interactions)
        # val_loader = FastTensorDataLoader(*valdata,
        #                                 batch_size=batch_size,
        #                                 shuffle=True)
        # print('down val_loader')
        # for batch_labels,batch_index1,batch_index2 in val_loader:
        #     bt +=1
        #     bar.update(1)
        #     batch_labels = torch.squeeze(batch_labels)

        #     batch_features1 = torch.stack([train_esm_data[i.item()] for i in batch_index1])
        #     batch_features1 = batch_features1.cuda()
        #     batch_features2 = torch.stack([train_esm_data[i.item()] for i in batch_index2])
        #     batch_features2 = batch_features2.cuda()
        #     batch_labels = batch_labels.cuda()
            
        # #     # output = net(batch_features1,batch_features2)
        # traindata = get_traindata(train_proteins, train_esm_data,train_pairs,notneg_interactions)
        # train_loader = FastTensorDataLoader(*traindata,
        #                                     batch_size=batch_size,
        #                                     shuffle=True)
        #         Val!=
        # batch_features1=torch.Size([64, 2560])
        # batch_features2=torch.Size([64, 2560])
        # print('down train_loader')
        
        # for batch_features1, batch_features2, batch_labels in train_loader:
        #     # bt +=1
        #     # bar.update(1)
        #     batch_labels = torch.squeeze(batch_labels)
        #     # batch_features1 = batch_features1.cuda()
        #     # batch_features2 = batch_features2.cuda()
        #     # batch_labels = batch_labels.cuda()
        #     print('Train!=')
        #     # output = net(batch_features1,batch_features2)
        #     print('batch_features1={}'.format(batch_features1.shape))
        #     print('batch_features2={}'.format(batch_features2.shape))
        #     #             Train!=
        #     # batch_features1=torch.Size([64, 2560])
        #     # batch_features2=torch.Size([64, 2560])
        #     break
        # print('eval load data done...',datetime.datetime.now(),flush=True)
        # print('works!!!!!!!!',flush=True)
        # break
        ########
        
        ##############################
        ####### initial model ########
        ##############################
        
        print('Initializing model...',datetime.datetime.now(),flush=True)

        if modelname =='one':
            outdim = indim
            net = Featuredotmodel(in_dim=indim, hidden_dim=hiddim, out_dim=outdim)
            print('Train model with only #one# protein modify')
            print('Model using dot product,indim={},hiddim={},outdim={}'.format(indim,hiddim,outdim),flush=True)
            print('Epochs-{},LR-{}'.format(epochs,learningrate),flush=True)
            print('BatchSize-{}'.format(batch_size),flush=True)
        elif modelname =='two':
            net = twoFeaturedotmodel(in_dim=indim, hidden_dim=hiddim, out_dim=outdim)
            print('Train model with only #two# protein modify')
            print('Model using dot product,indim={},hiddim={},outdim={}'.format(indim,hiddim,outdim),flush=True)
            print('Epochs-{},LR-{}'.format(epochs,learningrate),flush=True)
            print('BatchSize-{}'.format(batch_size),flush=True)

        # loss_func = PUloss
        loss_func=tri_PUloss
        print('loss function = tri_PUloss, and with 10 time pos loss ',flush=True)
        print('torch.mean(pos) - torch.mean(torch.mean(notneg) + torch.mean(neg)) +5',flush=True)
        sig = torch.nn.Sigmoid()
        optimizer = torch.optim.Adam(net.parameters(), lr=learningrate)
        scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)
        print('>>>model setting done',datetime.datetime.now(), flush=True)

        ######################
        ####### train ########
        ######################
        ## save best model at following path
        best_model_file = f'{data_root}/model_{modelname}{indim}_{outdim}_e{epochs}_bs{batch_size}.th'

        ### train loader
        print('>>>Loading traindata ....',datetime.datetime.now(), flush=True)
        trainp = f'{data_root}/train_fold{fold}.pkl'
        valp = f'{data_root}/val_fold{fold}.pkl'

        # try:
        #     traindata = pd.read_pickle(trainp)
        # except:
        #     print('>>>Loading traindata done',datetime.datetime.now(), flush=True)
        #     # traindata = get_traindata(train_proteins, train_esm_data,train_pairs,notneg_interactions)
        #     traindata = get_index_val_testdata(train_proteins, train_esm_data,train_pairs,notneg_interactions)
        #     pd.to_pickle(traindata,trainp)

        # try:
        #     valdata = pd.read_pickle(valp)
        # except:
        #     print('>>>Loading valdata',datetime.datetime.now(), flush=True)
        #     valdata = get_index_val_testdata(val_proteins, train_esm_data,valid_pairs,notneg_interactions)
        #     pd.to_pickle(traindata,trainp)
        
        valdata = get_index_val_testdata(val_proteins, train_esm_data,valid_pairs,notneg_interactions)
        # traindata = get_index_val_testdata(train_proteins, train_esm_data,train_pairs,notneg_interactions,ktime=100)
        # print('ktime =100')
        traindata = get_index_val_testdata(train_proteins, train_esm_data,train_pairs,notneg_interactions)
        print('ktimeget_index_val_testdata=100')
        # traindata = get_index_val_testdata(train_proteins, train_esm_data,train_pairs,notneg_interactions)

        train_loader = FastTensorDataLoader(*traindata,
                                            batch_size=batch_size,
                                            shuffle=True)
       
        ### val loader
        val_loader = FastTensorDataLoader(*valdata,
                                            batch_size=batch_size,
                                            shuffle=True)
        print('eval load data done...',datetime.datetime.now(),flush=True)

        net.train()
        if torch.cuda.is_available():
            net = net.cuda()
            print("Transferred model to GPU")
        # print('Training the model from the begnning', flush=True)
        print('Training the model from the begnning...',datetime.datetime.now(),flush=True)

        best_loss = 999999999.0
        
        for epoch in range(epochs):
            print('Epoch={}...'.format(epoch),datetime.datetime.now(),flush=True)
            bt=0
            train_loss = 0
            train_steps = int(math.ceil(len(traindata[0]) / batch_size))
            # print('@@@@@@@if 1. not in batch_labels:   continue',flush=True)
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_labels,batch_index1,batch_index2 in train_loader:
                    bt +=1
                    bar.update(1)
                    # if 1. not in batch_labels:
                    #     continue
                    batch_labels = torch.squeeze(batch_labels)
                    
                    batch_features1 = torch.stack([train_esm_data[i.item()] for i in batch_index1])
                    batch_features1 = batch_features1.cuda()
                    batch_features2 = torch.stack([train_esm_data[i.item()] for i in batch_index2])
                    batch_features2 = batch_features2.cuda()
                    batch_labels = batch_labels.cuda()
                    output = net(batch_features1,batch_features2)
                    # pos1 not neg 2 unlabel3
                    try:
                        loss = loss_func(output, batch_labels)
                    except:
                        print('lossbug===',flush= True)
                        continue
                        # batch_labels = torch.unsqueeze(batch_labels, 0)
                        # loss = loss_func(output, batch_labels)

                # for batch_features1, batch_features2, batch_labels in train_loader:
                #     bt +=1
                #     bar.update(1)
                #     batch_labels = torch.squeeze(batch_labels)
                #     batch_features1 = batch_features1.cuda()
                #     batch_features2 = batch_features2.cuda()
                #     batch_labels = batch_labels.cuda()
                #     output = net(batch_features1,batch_features2)
                #     try:
                #         loss = loss_func(output, batch_labels)
                #     except:
                #         loss = loss_func(output, batch_labels)
                    total_loss = loss
                    train_loss += loss.detach().item()
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    if bt % 1250==0:
                        print('batch loss=', total_loss.item(), flush=True)
                    # if bt % 200 == 0 and epoch >= 25:
                    #     # print('batch loss=', total_loss.item(), flush=True)
                    #     print('ouput={}\nbatch_labels={}|{}'.format(
                    #         output[:5], batch_labels[:5],total_loss.item()))  
                print('!!?bt={}'.format(bt))          
            train_loss/= train_steps


            ###########################
            ####### evaluation ########
            ###########################

            step = 5
            # if epoch % step == 0 and epoch >1:
            if epoch % step == 0 :
                net.eval()  
                bt =0
                with torch.no_grad():
                    valid_steps = int(math.ceil(len(valdata[0]) / batch_size))
                    valid_loss = 0
                    preds = []
                    alllabel = []
                    p1 = []
                    p2 = []
                    for batch_labels,batch_index1,batch_index2 in val_loader:
                        bt +=1
                        bar.update(1)
                        batch_labels = torch.squeeze(batch_labels)
                        batch_features1 = torch.stack([train_esm_data[i.item()] for i in batch_index1])
                        batch_features1 = batch_features1.cuda()
                        batch_features2 = torch.stack([train_esm_data[i.item()] for i in batch_index2])
                        batch_features2 = batch_features2.cuda()
                        batch_labels = batch_labels.cuda()
                        output = net(batch_features1,batch_features2)
                        # pos1 not neg 2 unlabel3
                        batch_loss = loss_func(output, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        if bt % 5000 ==0:
                            # print('output={}'.format(output[:5]),flush =True)
                            # print('batch_labels={}'.format(batch_labels[:5]),flush =True)
                            print('val loss={}'.format(batch_loss),flush =True)
                    # for batch_features1, batch_features2, batch_labels,batch_index1,batch_index2 in val_loader:
                    #     bt +=1
                    #     bar.update(1)
                    #     batch_labels = torch.squeeze(batch_labels)
                        
                    #     batch_features1 = batch_features1.cuda()
                    #     batch_features2 = batch_features2.cuda()
                    #     batch_labels = batch_labels.cuda()

                    #     output = net(batch_features1,batch_features2)
                    #     # pos1 not neg 2 unlabel3
                    #     batch_loss = loss_func(output, batch_labels)
                    #     valid_loss += batch_loss.detach().item()
                        preds = np.append(
                                        preds,
                                        output.detach().cpu().numpy())
                        alllabel = np.append(
                                        alllabel,
                                        batch_labels.detach().cpu().numpy()) 
                        p1 = np.append(
                                        p1,
                                        batch_index1.detach().cpu().numpy())
                        p2 = np.append(
                                        p2,
                                        batch_index2.detach().cpu().numpy())
                        print('val !!?bt={}'.format(bt))          
                        
                    print('eval preds done...',datetime.datetime.now(),flush=True)
                        
                    predresult = pd.DataFrame({
                        "p1":p1,
                        "p2":p2,
                        # "pairs": alltestpairs,
                        "labels": alllabel,
                        "preds":preds
                        })
                    print('eval calulate ranks...',datetime.datetime.now(),flush=True)
                    
                    ###############
                    ranks,n_prots = compute_rank(predresult)
                    Mean_avg_precision = compute_MAP(predresult)
                    rk_roc = compute_rank_roc(ranks, n_prots)

                    
                    # bi_alllabel = alllabel[(alllabel == 1) | (alllabel == 0)]
                    # bi_preds = preds[(alllabel == 1) | (alllabel == 0)]

                    ## tri
                    bi_alllabel = alllabel[(alllabel == 1) | (alllabel == 3)]
                    bi_preds = preds[(alllabel == 1) | (alllabel == 3)]      
                    bi_alllabel[bi_alllabel == 3] = 0

                    bi_preds = sig(torch.from_numpy(bi_preds))
                    print(':4 bi_alllabele={},bipreds={}'.format(bi_alllabel[:4],bi_preds[:4]),flush=True)
         
                    _, _, _, roc_auc = compute_roc(bi_alllabel, bi_preds)

                    valid_loss /= valid_steps
                    
                    print(f'Epoch {epoch}: Loss - {train_loss},Valid loss - {valid_loss}, \nrkAUC - {rk_roc},AUC - {roc_auc}, MAP - {Mean_avg_precision}',flush=True)     

                    # print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, rkAUC - {rk_roc},AUC - {roc_auc}',flush=True)     

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model', flush=True)
                    torch.save(net.state_dict(), best_model_file) 
                else:
                    print('not Saving model',best_loss,valid_loss, flush=True)

                scheduler.step()
                        
        ######################
        ####### test  ########
        ######################

        # Load the best model 
        print('..Loading the best model')
        print('Loading the best model FOR test...',datetime.datetime.now(),flush=True)

        net.load_state_dict(torch.load(best_model_file))
        net.eval()
        print('Loading test data..',datetime.datetime.now(),flush=True)

        testdata = get_index_val_testdata(test_prot_index, test_esm_data,test_pairs,notneg_interactions)
        test_loader = FastTensorDataLoader(*testdata,
                                            batch_size=batch_size,
                                            shuffle=True)
        print('Loaded test data',datetime.datetime.now(),flush=True)

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[0]) / batch_size))
            test_loss = 0
            preds = []
            alllabel = []
            p1 =[]
            p2=[]
            for batch_labels,batch_index1,batch_index2 in test_loader:
                bt +=1
                bar.update(1)
                batch_labels = torch.squeeze(batch_labels)

                batch_features1 = torch.stack([test_esm_data[i.item()] for i in batch_index1])
                batch_features1 = batch_features1.cuda()
                batch_features2 = torch.stack([test_esm_data[i.item()] for i in batch_index2])
                batch_features2 = batch_features2.cuda()
                batch_labels = batch_labels.cuda()
                output = net(batch_features1,batch_features2)
                # pos1 not neg 2 unlabel3
                batch_loss = loss_func(output, batch_labels)
                test_loss += batch_loss.detach().item()
            # for batch_features1, batch_features2, batch_labels,batch_index1,batch_index2 in test_loader:
            #     bt +=1
            #     bar.update(1)
            #     batch_labels = torch.squeeze(batch_labels)

            #     batch_features1 = batch_features1.cuda()
            #     batch_features2 = batch_features2.cuda()
            #     batch_labels = batch_labels.cuda()
                
            #     output = net(batch_features1,batch_features2)
            #     # pos1 not neg 2 unlabel3
            #     batch_loss = loss_func(output, batch_labels)
            #     test_loss += batch_loss.detach().item()
                preds = np.append(
                                preds,
                                output.detach().cpu().numpy())
                alllabel = np.append(
                                alllabel,
                                batch_labels.detach().cpu().numpy()) 
                p1 = np.append(
                                p1,
                                batch_index1.detach().cpu().numpy())
                p2 = np.append(
                                p2,
                                batch_index2.detach().cpu().numpy())
            test_loss /= test_steps    
        predresult = pd.DataFrame({
            "p1":p1,
            "p2":p2,
            # "pairs": alltestpairs,
            "labels": alllabel,
            "preds":preds
            })
        print('test preds make list done',datetime.datetime.now(),flush=True)
        
        ###############
        out_file = f'{data_root}/test_predsresults_{modelname}{indim}_{outdim}_e{epochs}_bs{batch_size}.pkl'
        predresult.to_pickle(out_file)
        print('test preds saved',datetime.datetime.now(),flush=True)

        ranks,n_prots = compute_rank(predresult)
        Mean_avg_precision = compute_MAP(predresult)
        rk_roc = compute_rank_roc(ranks, n_prots)
        eval_file = f'{data_root}/test_E_{modelname}{indim}_{outdim}_e{epochs}_bs{batch_size}.txt'
        
        print(f'Test Loss - {test_loss}, rkAUC - {rk_roc}, MAP - {Mean_avg_precision}', flush=True)
        with open(eval_file,'w+') as f:
            print(f'Test Loss - {test_loss}, rkAUC - {rk_roc}',file= f, flush=True)
            print(f'>>>>>>n_prots - {n_prots}',file= f, flush=True)
            print(f'>>>>>>Ranks - \n{ranks}',file= f, flush=True)
        
        
        ######################
        ####### print ########
        ######################

        print('out_file ={}'.format(out_file))
        print('eval_file ={}'.format(eval_file))
        print('best_model_file ={}'.format(best_model_file))
        print('interactions_index_fp ={}'.format(interactions_index_fp))
        print('notneg_interactions_index_fp ={}'.format(notneg_interactions_index_fp))
        print('posproteins_index_fp ={}'.format(posproteins_index_fp))
        print('proteins_fp ={}'.format(proteins_fp))
        print('esm_fp ={}'.format(esm_fp))
        
      
if __name__ == '__main__':

    main()










    
