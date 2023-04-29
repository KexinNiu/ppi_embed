from loaddata0216 import Loaddata
# printf 'Based on 0206 metric.py'
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
from network import Featuredotmodel, twoFeaturedotmodel,PUloss
from utils import topk, acc_thre, aupr_thre, compute_aupr, compute_roc
from utils_data import get_data, get_test_data

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
    default='dot',
    help='options =')
# ['dot','Euclidean distance','Manhattan metric','Chebychev metric']
@ck.option(
    '--learningrate', '-lr', default=1e-4,
    help='learning rate')
def main(data_root, batch_size, epochs, hiddim, load, device, negativefold, name, esmfolder, modelname, learningrate, metric):
    print('>>>', torch.cuda.is_available())
    lr = learningrate
    print('lr=', lr, flush=True)
    print('batch_size=', batch_size, flush=True)

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
        print('>>>!!not exists', flush=True)
        dataset = Loaddata(high_ppipath, low_ppipath,
                           esmfolder, svfolder, negativefold, name)
        allinfo = dataset.allinfo
        allpairs = dataset.allpairs
        name2index = dataset.name2index
        index2name = dataset.index2name
        print('>>>sv...', flush=True)
        print('>>>finished sv', flush=True)

    else:

        print('\n\n >>>load...', flush=True)
        allinfo = pd.read_pickle(allinfo_file)
        allpairs = pd.read_csv(allpairs_csvfile)
        allpairs = allpairs.astype(int)
        with open(name2index_file, 'r')as f:
            name2index = json.load(f)

        print('>>>finished loading\n', flush=True)

    if modelname == 'onef':
        if metric == 'dot':
            net = Featuredotmodel(in_dim=1280, hidden_dim=hiddim, out_dim=1280)

    elif modelname == 'twof':
        if metric == 'dot':
            net = twoFeaturedotmodel(in_dim=1280,
                                     hidden_dim=hiddim,
                                     out_dim=1280)

    loss_func = PUloss
    
    # lr = lr
    print('lr=', lr, flush=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)
    print('>>>model setting done\n', flush=True)

    # model
    ########################################

    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    for i, (train_index, test_index) in enumerate(kf.split(allinfo)):
        print(f"Fold {i}:", flush=True)
        valid_index = train_index[:round(len(train_index) * 0.25)]
        train_index1 = list(
            set(train_index).difference(
                set(train_index[:round(len(train_index) * 0.25)])))

        train_data = allinfo.iloc[train_index1]
        test_data = allinfo.iloc[test_index]

        valid_data = allinfo.iloc[valid_index]

        train_pairs = allpairs[(allpairs['p1'].isin(train_index1))
                               & (allpairs['p2'].isin(train_index1))]
        test_pairs = allpairs[(allpairs['p1'].isin(test_index))
                              & (allpairs['p2'].isin(test_index))]
        valid_pairs = allpairs[(allpairs['p1'].isin(valid_index))
                               & (allpairs['p2'].isin(valid_index))]

        # test_data
        # print('>>>test_data:\n',test_data[:5],flush=True)
        # print('>>>>>>>>>>>>>:\n',flush=True)
        # print('>>>test_pairs:\n',test_pairs[:5],flush=True)
        # print('>>>>>>>>>>>>>:\n',flush=True)

        # break

        traininfo_file = f'{data_root}/{name}_{i}_traininfo.pkl'
        trainpairs_file = f'{data_root}/{name}_{i}_trainpairs.pkl'

        train_data.to_pickle(traininfo_file)
        train_pairs.to_pickle(trainpairs_file)

        traindata = get_data(train_data, train_pairs)
        train_loader = FastTensorDataLoader(*traindata,
                                            batch_size=batch_size,
                                            shuffle=True)

        validdata = get_data(valid_data, valid_pairs)

        valid_loader = FastTensorDataLoader(*validdata,
                                            batch_size=batch_size,
                                            shuffle=True)
        # data prepare##
        print(' # data prepare##', flush=True)
        #####################################################

        best_loss = 100000.0
        if not load:
            net.train()
            print('Training the model from the begnning', flush=True)
            for epoch in range(epochs):
                train_loss = 0
                pt = 0
                train_steps = int(math.ceil(len(train_data) / batch_size))
                with ck.progressbar(length=train_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in train_loader:
                        bar.update(1)
                        pt += 1
                        batch_labels = torch.squeeze(batch_labels)
                        output = net(batch_features)
                        try:
                            loss = loss_func(output, batch_labels)
                        except:
                            loss = loss_func(output, batch_labels)
                        total_loss = loss
                        train_loss += loss.detach().item()
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()

                        if pt % 500 == 0:
                            print('batch loss=', total_loss.item(), flush=True)
                        if pt % 2500 == 0 and epoch >= 25:
                            print('ouput={}\nbatch_labels={}'.format(
                                output[:5], batch_labels[:5]))
                train_loss /= train_steps
                print('>>Validation', flush=True)
                net.eval()
                with torch.no_grad():
                    valid_steps = int(math.ceil(len(valid_data) / batch_size))
                    valid_loss = 0
                    preds = []
                    alllabel = []

                    with ck.progressbar(length=valid_steps,
                                        show_pos=True) as bar:
                        for batch_features, batch_labels in valid_loader:
                            bar.update(1)

                            batch_labels = torch.squeeze(batch_labels)
                            logits = net(batch_features)
                            batch_loss = loss_func(logits, batch_labels)

                            # batch_loss = F.binary_cross_entropy(logits, batch_labels)
                            valid_loss += batch_loss.detach().item()
                            preds = numpy.append(preds,
                                                 logits.detach().cpu().numpy())
                            alllabel = numpy.append(
                                alllabel,
                                batch_labels.detach().cpu().numpy())

                    valid_loss /= valid_steps
                    fpr, tpr, thresholds, roc_auc = compute_roc(
                        alllabel, preds)

                    print(
                        f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}',
                        flush=True)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model', flush=True)
                    torch.save(net.state_dict(), model_file)

                scheduler.step()

        # Loading best model
        print('..Loading the best model')
        net.load_state_dict(torch.load(model_file))
        net.eval()
        ####
        # testdata = get_data(test_data,test_pairs,True)
        testdata = get_test_data(test_data, test_pairs, True)
        test_loader = FastTensorDataLoader(*testdata,
                                           batch_size=batch_size,
                                           shuffle=True)

        ####

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[1]) / batch_size))
            test_loss = 0
            preds = []
            alllabel = []
            alltestpairs = []

            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels, bathch_indexs in test_loader:
                    bar.update(1)

                    batch_labels = torch.squeeze(batch_labels)
                    logits = net(batch_features)
                    batch_loss = loss_func(logits, batch_labels)

                    # batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = numpy.append(preds, logits.detach().cpu().numpy())
                    alllabel = numpy.append(
                        alllabel,
                        batch_labels.detach().cpu().numpy())

                    x0index, x1index = bathch_indexs.chunk(2, axis=1)
                    x0index = torch.squeeze(x0index).tolist()
                    x1index = torch.squeeze(x1index).tolist()
                    pp = zip(x0index, x1index)
                    for p in pp:
                        alltestpairs.append(p)

                test_loss /= test_steps

            fpr, tpr, thresholds, roc_auc = compute_roc(alllabel, preds)

            print(f'Test Loss - {test_loss}, AUC - {roc_auc}', flush=True)

        predresult = pd.DataFrame({
            "pairs": alltestpairs,
            "labels": alllabel,
            "preds": preds
        })
        predresult.to_pickle(out_file)
        eval_ranking(out_file, eval_file)


def eval_ranking(predp, outp):
    outf = open(outp, 'w')
    # p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
    predresult = pd.read_pickle(predp)
    predresult['protein1'] = predresult['pairs'].map(lambda x: int(x[0]))
    predresult['protein2'] = predresult['pairs'].map(lambda x: int(x[1]))
    aa = predresult.sort_values(by=['protein1', 'protein2'])
    # print('>>>>>>\n',aa[:5])
    unique_proteins = numpy.unique(aa[["protein1", "protein2"]].values)
    a1_list = []
    a10_list = []
    n = 0
    print('total={}'.format(len(unique_proteins)))
    for protein in unique_proteins:
        n += 1
        if n % (len(unique_proteins) // 10) == 0:
            print('>>{}/{}'.format(n, len(unique_proteins)))
        allinters = aa.query("protein1==@protein | protein2==@protein")
        allinters = allinters.sort_values(by=['preds'], ascending=False)
        # negvals = allinters.query('labels==0.0')['preds']
        posvals = allinters.query('labels==1.0')['preds']
        # print('posvals=\n',posvals)
        if len(posvals) >= 1:
            # print('>>>>>all top10\n',allinters.sort_values(by=['preds'],ascending=False)[:10])
            at1 = allinters['labels'].values[0]

            if len(posvals) >= 10:
                top10 = allinters[:10]
                at10 = sum(top10['labels'])
                a1_list.append(at1)
                a10_list.append(at10)
                print('{} \t @1={} @10={}'.format(protein, at1, at10),
                      file=outf)
            else:
                print('{} \t @1={} @10=None'.format(protein, at1),
                      file=outf,
                      flush=True)

    print('a1_list\ntotal={},hit@1={}'.format(len(a1_list),
                                              sum(a1_list) / len(a1_list)),
          file=outf)
    print('a10_list\ntotal={},hit@10={}'.format(
        len(a10_list),
        sum(a10_list) / (10 * len(a10_list))),
          file=outf)


if __name__ == '__main__':

    main()
