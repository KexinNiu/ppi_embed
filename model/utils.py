from collections import deque, Counter

from xml.etree import ElementTree as ET
import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_curve, auc, matthews_corrcoef,precision_recall_curve

from sklearn.metrics import PrecisionRecallDisplay
import statistics
import matplotlib.pyplot as plt
import sys

class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def get_anchestors(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()

        for term_id in terms:
            prop_terms |= self.get_anchestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set


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
        predr = np.where(preds >= thre,1,0)
        accurancy = sum(predr)/len(predr)
        acc.append(accurancy)
        maxindex = np.argmax(np.array(acc))
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
    print('labels={}'.format(labels))
    print('preds={}'.format(preds),flush=True)
    
    fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return fpr,tpr,thresholds,roc_auc

def compute_aupr(labels,preds):
    precision, recall, thresholds = precision_recall_curve(labels.flatten(), preds.flatten())
    print('len thre,pre,recall = {}|{}|{}'.format(len(thresholds),len(precision),len(recall)),flush=True)
    
    return precision,recall,thresholds

def compute_rank(predresult):
    pd.options.mode.chained_assignment = None
    '''
    predresult = pd.DataFrame({
                "p1":p1,
                "p2":p2,
                "labels": alllabel,
                "preds":preds
                })
    '''

    ranks={} ## key = protein, val
    uniprotiein = np.unique(predresult[["p1", "p2"]].values)
    n = len(uniprotiein)
    cn = n
    print('nnnn ={}'.format(n),flush=True)
    mean_rank = 0
    totalmeanrank =0
    top1,top10,top100 =0,0,0
    for protein in uniprotiein:
        ctop1,ctop10,ctop100,cmean_rank =0,0,0,0
        # protein_preds = predresult.query("p1==@protein | p2==@protein")
        protein_preds = predresult.query("p1==@protein")
        # print('len pro_preds={}'.format(len(protein_preds)))
        rkll = rankdata(protein_preds['preds'].values, method='average')
        # print('len rkll={}'.format(len(rkll)),flush=True)
        protein_preds[ 'rankval'] = rkll
        # protein_preds['rankval']= rankdata(protein_preds['preds'], method='average')
        pos_rankvals = protein_preds.query('labels==1.0')['rankval']
        # print('len pos_rankvals={}'.format(len(pos_rankvals)),flush=True)

        # pos_rankvals 排第x位 pos-rankvals < uniproteins 的数量
        for rankval in pos_rankvals:
            if rankval == 1:
                top1 += 1
            if rankval <= 10:
                top10 += 1
            if rankval <= 100:
                top100 += 1
            cmean_rank +=rankval
            mean_rank += rankval
            if rankval not in ranks:
                ranks[rankval] = 0
            ranks[rankval] += 1
        
        # print('ranks:lenk{},{}'.format(len(ranks.keys()),ranks),flush=True)
        try:
            cmean_rank /= len(pos_rankvals)
        except:
            cn -=1
            cmean_rank = 0
        totalmeanrank += cmean_rank
    print('ranks keys={}'.format(list(ranks.keys())[:10]),flush=True)
        

    top1 /= n
    top10 /= n
    top100 /= n
    mean_rank /= n
    totalmeanrank /=n 
    

    print('top1={},top10={},top100={},mean_rank={},totalmeanrk={}'.format(top1,top10,top100,mean_rank,totalmeanrank),flush=True)
    # print('ranks keys'.format(list(ranks.keys())),flush=True)
    return ranks,len(uniprotiein)

def compute_rank_roc(ranks, n_prots):
    auc_x = list(ranks.keys())
    auc_x.sort()
    auc_y = []
    tpr = 0
    sum_rank = sum(ranks.values())
    for x in auc_x:
        tpr += ranks[x]
        auc_y.append(tpr / sum_rank)
    auc_x.append(n_prots)
    auc_y.append(1)
    # print('auc_y|{}'.format(auc_y))
    # print('auc_x|{}'.format(auc_x))
    auc = np.trapz(auc_y, auc_x) / n_prots
    # print('auc={}'.format(auc))
    # print('change auc={}'.format(np.trapz(auc_x, auc_y) / n_prots))
    # auc=0.16795170030379
    # change auc=0.8320100109455038
    return auc


def computeAVP(predresult,protein,averagePrecision):
    # https://github.com/DavidRFerreira/InformationRetrieval_EvaluationMetrics/blob/main/src/computeMetrics.py
    pd.options.mode.chained_assignment = None
    precisionlist=[]
    recalllist=[]
    totalRelevantAlreadyFound = 0
    protein_preds = predresult.query("p1==@protein")
    rkll = rankdata(protein_preds['preds'].values, method='average')
    protein_preds[ 'rankval'] = rkll

    sort_preds = protein_preds.sort_values(by=['rankval'])
    pos_rankvals = protein_preds.query('labels==1.0')['rankval']
    numTotalRelevant = len(pos_rankvals)
    if numTotalRelevant ==0:
        return precisionlist,recalllist,averagePrecision,numTotalRelevant

    # print('len(sort/preds',len(sort_preds))
    for i, row in sort_preds.iterrows():
    # for i in range(0,len(sort_preds)):
    #     print('i',i)
    #     print('sort preds =',sort_preds)
    #     print('1-',sort_preds[i]['labels'])
    #     print('1-',sort_preds[i]['labels']==1)
    #     print('2-',sort_preds.at[i,'labels'])
    #     print('2-',sort_preds.at[i,'labels']==1)
        if row['labels'] == 1:
            totalRelevantAlreadyFound += 1
            averagePrecision = averagePrecision + totalRelevantAlreadyFound / (i + 1)    
        precisionlist.append(totalRelevantAlreadyFound/(i+1))
        recalllist.append(totalRelevantAlreadyFound/numTotalRelevant)
    
    averagePrecision = averagePrecision/ totalRelevantAlreadyFound
    # print("\nSET:")
    # print(set)
    # print("\nPRECISION:")
    # print(precisionlist)
    # print("\nRECALL:")
    # print(recalllist)
    # print("\nAVERAGE PRECISION:")
    # print(averagePrecision)
    # print("")
    return precisionlist,recalllist,averagePrecision,numTotalRelevant

def plotMultiplePrecisionRecallCurve(precision, recall):
    _, ax = plt.subplots(figsize=(7, 8))
    colors = ["navy", "darkorange", "teal", "red", "bisque", "olive", "lavender"]

    for idx, prec in enumerate(precision):
        disp = PrecisionRecallDisplay(precision[idx], recall[idx])
        label = "Precision-recall for query #" + str(idx+1)
        disp.plot(ax=ax, name=label, color=colors[idx])
    
    plt.legend(loc='lower right')
    plt.savefig('precisionRecall.pdf')

def compute_MAP(predresult):
    uniprotiein = np.unique(predresult[["p1", "p2"]].values)
    n = len(uniprotiein)
    precisionSet=[]
    recallSet = []
    averagePrecision =0
    avpSet=[]
    for protein in uniprotiein:
        precision,recall,averagePrecision,numTotalRelevant = computeAVP(predresult,protein,averagePrecision)
        if numTotalRelevant ==0:
            continue
        avpSet.append(averagePrecision)
        precisionSet.append(precision)
        recallSet.append(recall)
    MAP = statistics.mean(avpSet)
    
    print("\nMAP:")
    print(MAP)
    # plotMultiplePrecisionRecallCurve(precisionSet, recallSet)
    return MAP

