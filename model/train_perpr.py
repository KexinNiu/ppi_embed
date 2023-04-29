## train per protein
## split by proteins
## negatives generates by random 
## negatives are not shown in low conf 

import torch
import numpy
from torch import nn
from simdataset import dataset_prosplit
import torch.utils.data as Data
from torchvision.models import *
import time
import random
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

class Featuremodel(nn.Module):
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

# ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.final_100.txt'
# ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.finalpairs.txt'

high_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.highconf.links.txt'
low_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.lowconf.links.txt'
# esmfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
esmfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/'

svfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/'

TRAIN_yeast = dataset_prosplit(allnames='',allpairs='',highppipath=high_ppipath,\
                                lowppipath=low_ppipath,esmfolder=esmfolder,\
                                svfolder=svfolder,datalabel='train')
allnames = TRAIN_yeast.allnames
allpairs = TRAIN_yeast.allpairs
notneg_pairs = TRAIN_yeast.notneg

n2f = TRAIN_yeast.n2f

trainposdd = TRAIN_yeast.posdd

VAL_yeast = dataset_prosplit(allnames=allnames,allpairs=allpairs,highppipath=high_ppipath,\
                                lowppipath=low_ppipath,esmfolder=esmfolder,\
                                svfolder=svfolder,datalabel='val')
valposdd = VAL_yeast.posdd

# TEST_yeast = dataset_prosplit(allnames=allnames,allpairs=allpairs,highppipath=high_ppipath,\
#                                 lowppipath=low_ppipath,esmfolder=esmfolder,\
#                                 svfolder=svfolder,datalabel='val')
# testposdd = TEST_yeast.posdd

##############batchloss define
def random_neg(notneg_pairs,pr,prs,kk):
    negp = []
    while len(negp) < kk:
        if len(prs)< kk:
            for curpr in prs:
                if curpr<pr and (curpr,pr) not in notneg_pairs:
                    negp.append(curpr)
                elif curpr>pr and (pr,curpr) not in notneg_pairs:
                    negp.append(curpr)
            return negp
            
        lack = kk - len(negp)
        # print('prs={},lack={}'.format(len(prs),lack))
        currprlist = random.sample(prs,lack)
        for curpr in currprlist:
            if curpr in negp:
                continue
            if curpr<pr and (curpr,pr) not in notneg_pairs:
                negp.append(curpr)
            elif curpr>pr and (pr,curpr) not in notneg_pairs:
                negp.append(curpr)
    return negp
def batchloss(n2f,ddpos,prs,notneg_pairs):

        # print('1type ddpos=',type(ddpos))

        cos_sim_pos =[]
        cos_sim_neg =[]

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        kk = 10
        for pr in prs:
            f0 = n2f[pr]
            f0 = torch.as_tensor(n2f[pr])
            try:
                pos_p = ddpos[pr]
            except:
                continue
            neg_p = random_neg(notneg_pairs,pr,prs,kk)
            
            for pp in pos_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_pos.append(output)

            for pp in neg_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_neg.append(output)


        cos_sim_pos = torch.tensor(cos_sim_pos,requires_grad=True)
        cos_sim_neg = torch.tensor(cos_sim_neg,requires_grad=True)
        posloss = 1 -  torch.mean(cos_sim_pos)
        negloss =   torch.mean(cos_sim_neg)
        totalloss = torch.add(posloss,negloss)
        return totalloss
##############
##############validataion
def validation(net,n2f,VAL_yeast,Vallabel='val',onlyloss = True):
    ## during
    ##      return total loss for all valproteins
    ## model eval
    ##      return file with aupr auroc microf1...
    BATCH_SIZE=100
    print('##### \nevaluation....',flush=True)
    valloader = Data.DataLoader(dataset=VAL_yeast,
                            shuffle=True,
                            batch_size=BATCH_SIZE
                        )
    # VAL_yeast.proteins
    # VAL_yeast.posdd
    ## 生成新的embeddings for all proteins
    

    for i,data in enumerate(valloader,0):
        # print('BATCH_SIZE=',BATCH_SIZE)
        # print('data=',data)
        for pr in data:
            # print('prname=',pr)
            orif = torch.as_tensor(n2f[pr])
            newf = net(orif)
            n2f[pr] = newf

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    
    if Vallabel =='val':
        # val eval
        ddpos=VAL_yeast.posdd
        prs = VAL_yeast.proteins

        ####
        prs = prs[:100]

        ####
        print('total....{}'.format(len(prs)),flush=True)
        kk = 10
        cos_sim_pos =[]
        cos_sim_neg =[]
        # 每个蛋白和所有蛋白的sim
        # 排序用不同的thr 然后看多少interacts
        ## validation的loss最小
        cout=0
        for pr in prs:
            cout+=1
            if (cout% round(len(prs)/5)) == 0:
                print('....{}'.format(cout),flush=True)

            f0 = n2f[pr]
            f0 = torch.as_tensor(n2f[pr])
            
            try:
                pos_p = ddpos[pr]
            except:
                continue
            neg_p = random_neg(notneg_pairs,pr,prs,kk)
            
            for pp in pos_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_pos.append(output)
            for pp in neg_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_neg.append(output)

        cos_sim_pos = torch.tensor(cos_sim_pos,requires_grad=True)
        cos_sim_neg = torch.tensor(cos_sim_neg,requires_grad=True)

        posloss = 1 -  torch.mean(cos_sim_pos)
        negloss =   torch.mean(cos_sim_neg)
        totalloss = torch.add(posloss,negloss)
        if onlyloss == True:
            return totalloss
        ### print auroc什么的
        def gettotaleval(prs,pairs,cos,thr):
            print('thr = {}'.format(thr),flush=True)
            # totaldict={}
            totalauroc=[]
            totaltaupr=[]
            totalF1=[]
            for i in range (0,len(prs)):
                p1 = prs[i]
                p1sim = []
                p1labels =[]
                f1 = torch.as_tensor(n2f[p1])
                for j in range(i+1,len(prs)):
                    p2 = prs[j]
                    f2 = torch.as_tensor(n2f[p2])
                    output = cos(f1,f2)
                    p1sim.append(output.item())
                    if p1<p2:
                        if (p1,p2) in pairs:
                            p1labels.append(1)
                        else:
                            p1labels.append(0)
                    else:
                        if (p2,p1) in pairs:
                            p1labels.append(1)
                        else:
                            p1labels.append(0)
                if sum(p1labels) ==len(p1labels) or sum(p1labels) ==0:
                    continue
                
                predlabel=[]
                for score in p1sim:
                    if score>thr:
                        predlabel.append(1)
                    else:
                        predlabel.append(0)
                # print('len pred =',predlabel)
                # print('len p1labels =',p1labels)
                p1labels = numpy.asarray(p1labels) 
                predlabel = numpy.asarray(predlabel) 
                p1sim = numpy.asarray(p1sim) 
                
                
                try:
                    precision, recall, _ = metrics.precision_recall_curve(p1labels, p1sim)
                    aupr = metrics.auc(recall, precision)
                    f1 = metrics.f1_score(p1labels, predlabel)
                    score = metrics.roc_auc_score(p1labels, p1sim)
                    totalauroc.append(score)
                    totaltaupr.append(aupr)
                    totalF1.append(f1)
                except:
                    continue    
            totalauroc = numpy.asarray(totalauroc) 
            totaltaupr = numpy.asarray(totaltaupr) 
            totalF1 = numpy.asarray(totalF1) 
            print('avg auroc={}'.format(numpy.average(totalauroc)))
            print('avg aupr={}'.format(numpy.average(totaltaupr)))
            print('avg totalF1={}'.format(numpy.average(totalF1)),flush=True)
            print("______________")
            return 
                
        # cos = nn.CosineSimilarity(dim=0, eps=1e-6)     
        gettotaleval(prs,VAL_yeast.allpairs,cos,thr=0.5)

        return totalloss  
            


    
    elif Vallabel =='test':
        pass

    return

##############
def train_sim_perprotine(NUMBER_EPOCHS,evalstep,savestep,BATCH_SIZE,SAVEPATH,loadpath,New=True):
    print('SAVEPATH',SAVEPATH,flush = True)
    print('loadpath',loadpath,flush = True)
    vallosslist=[100]
    net  = Featuremodel(1280,200,1280)
    print('Loading model....',flush = True)
    print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
    net.train()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    print('lr=1e-3',flush = True)
    print('BATCH_SIZE={}'.format(BATCH_SIZE),flush = True)
    start_epoch =-1
    print('dataloader time=',time.asctime( time.localtime(time.time()) ), flush=True)
    Trainloader = Data.DataLoader(dataset=TRAIN_yeast,
                            shuffle=False,
                            batch_size = BATCH_SIZE
                            )

    # valloader = Data.DataLoader(dataset=VAL_yeast,
    #                         shuffle=True,
    #                         batch_size=BATCH_SIZE
    #                     )
    # VAL_yeast.proteins
    # VAL_yeast.posdd

    print('//time=',time.asctime( time.localtime(time.time()) ), flush=True)
    print('Train $ Test initial finish : ',flush=True)
    if New == False:
        print('loading check point...',flush = True)
        checkpoint = torch.load(loadpath)  # 加载断点
        net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch'] 
        print('start epoch ==',start_epoch,flush= True)

## neg by rand (neg50-100>pos1)skip interactions with low confident 
# eval by protein with rank cos sim and 
# split proteins 
            # if epoch % savestep ==0 and epoch !=0:
    count = 0
    for epoch in range(start_epoch+1,NUMBER_EPOCHS+1):
        print('epoch',epoch,'start.',flush=True)
        if epoch % evalstep==0 :
            
            net.eval()
            avgloss = validation(net,n2f,VAL_yeast,Vallabel='val',onlyloss=True)
            # avgloss = Validation(valloader)

            print('Val loss=',avgloss,flush=True)
            if avgloss>vallosslist[-1]:
                count+=1
            else:
                count==0
        print('time=',time.asctime(time.localtime(time.time())), flush=True)
        # torch.cuda.empty_cache()

        for i,data in enumerate(Trainloader,0):
            # print('BATCH_SIZE=',BATCH_SIZE)
            # print('data=',data)
            for pr in data:
                # print('prname=',pr)
                orif = torch.as_tensor(n2f[pr])
                newf = net(orif)
                n2f[pr] = newf
            bloss = batchloss(n2f,trainposdd,data,notneg_pairs)
            print('bloss:',bloss.item(),flush=True)
            bloss.backward()
            optimizer.step()  

        # if epoch % evalstep==0 and epoch >=evalstep:
        if epoch % evalstep==0 :
            
            net.eval()
            avgloss = validation(net,n2f,VAL_yeast,Vallabel='val',onlyloss=False)
            # avgloss = Validation(valloader)

            print('Val loss=',avgloss,flush=True)
            if avgloss>vallosslist[-1]:
                count+=1
            else:
                count=0

            if count ==3:
                print('No improving!',flush=True)
                print('avgloss=',avgloss,"epoch number = ",epoch,flush=True)
                checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch
                }
                print('save checkpoint..',flush = True)
                print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
                checkpointpath = SAVEPATH + '_simprosplit'+str(epoch)+'.pth'
                # checkpointpath ='/ibex/scratch/niuk0a/projects/dscript/PPI/PPI_ESM/model/trained_ckpt/ckpt_'+str(epoch)+'.pth'
                torch.save(checkpoint, checkpointpath)
                ##save model
                print('_____________________',flush=True)
                
                break
###########save
        # if epoch % savestep ==0 and epoch !=0:
        if epoch % savestep ==0:
            checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch
                }
            print('save checkpoint..',flush = True)
            print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
            checkpointpath = SAVEPATH + '_simprosplit'+str(epoch)+'.pth'
            # checkpointpath ='/ibex/scratch/niuk0a/projects/dscript/PPI/PPI_ESM/model/trained_ckpt/ckpt_'+str(epoch)+'.pth'
            torch.save(checkpoint, checkpointpath)

        
    checkpoint = {
                    "net": net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    "epoch": epoch
                }
    print('save checkpoint..',flush = True)
    print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
    checkpointpath = SAVEPATH + '_simprosplit'+str(epoch)+'.pth'
    # checkpointpath ='/ibex/scratch/niuk0a/projects/dscript/PPI/PPI_ESM/model/trained_ckpt/ckpt_'+str(epoch)+'.pth'
    torch.save(checkpoint, checkpointpath)

# train_sim_perprotine(NUMBER_EPOCHS=2,BATCH_SIZE=2,MODEL_SAVE='',SAVEPATH='',loadpath='',New=True)
svp= '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/svmodel/test12ep'
# train_sim_perprotine(NUMBER_EPOCHS=1,evalstep=1,savestep=1,BATCH_SIZE=8,SAVEPATH=svp,loadpath='',New=True)

# NUMBER_EPOCHS=1
# evalstep=1
# savestep=1
# BATCH_SIZE=8
# SAVEPATH=svp
# loadpath=''
# New=True
# print()
train_sim_perprotine(NUMBER_EPOCHS=12,evalstep=12,savestep=4,BATCH_SIZE=4,SAVEPATH=svp,loadpath='',New=True)



