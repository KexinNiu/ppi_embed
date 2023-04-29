import torch
from torch import nn
from simdataset import dataset_pairwise,dataset_perprotein
import torch.utils.data as Data
from torchvision.models import *
import time
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

ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.final_100.txt'
ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.finalpairs.txt'

esmfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
svfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/'

# TRAIN_yeast = dataset_pairwise(ppipath,esmfolder,svfolder,Train='train')
# allpairs = TRAIN_yeast.allpairs
# n2f = TRAIN_yeast.n2f
# ddpos_neg = TRAIN_yeast.ddpos_neg
# VAL_yeast = dataset_pairwise(allpairs,esmfolder,svfolder,Train='val')
# TEST_yeast = dataset(allpairs,esmfolder,svfolder,Train='test')


def train_sim_pairwise(NUMBER_EPOCHS,BATCH_SIZE,MODEL_SAVE,SAVEPATH,loadpath,New=True):
    print('SAVEPATH',SAVEPATH,flush = True)
    print('loadpath',loadpath,flush = True)
    # net  = Featuremodel(1280,100,200).cuda()
    print('Loading model....',flush = True)
    print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
    # net.train()
    # criterion = nn.BCELoss()
    # optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    print('lr=1e-3',flush = True)
    start_epoch =-1
    print('dataloader time=',time.asctime( time.localtime(time.time()) ), flush=True)
    Trainloader = Data.DataLoader(dataset=TRAIN_yeast,
                            shuffle=False,
                            batch_size = BATCH_SIZE
                            )
    # testloader = Data.DataLoader(dataset=VAL_yeast,
    #                         shuffle=True,
    #                         batch_size=1
    #                     )
    print('//time=',time.asctime( time.localtime(time.time()) ), flush=True)
    print('Train $ Test initial finish : ',flush=True)
    # if New == False:
    #     print('loading check point...',flush = True)
    #     checkpoint = torch.load(loadpath)  # 加载断点
    #     net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
    #     optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    #     start_epoch = checkpoint['epoch'] 
    #     print('start epoch ==',start_epoch,flush= True)


    for epoch in range(start_epoch+1,NUMBER_EPOCHS+1):
        print('epoch',epoch,'start.',flush=True)
        
        print('time=',time.asctime(time.localtime(time.time())), flush=True)
        # torch.cuda.empty_cache()
        for i,data in enumerate(Trainloader,0):
            print('BATCH_SIZE=',BATCH_SIZE)
            # print('data=',data)
            p1 = data[0][0]
            p2 = data[0][1]
            print('p1=',p1)
            print('p2=',p2)


            # batch_token1,batch_token2 = data[0][1],data[1][1]
            # batch_token1 = batch_token1.cuda()
            # batch_token2 = batch_token2.cuda()

            # ylabel = data[2]            
            # ylabel = ylabel.clone().detach()
            # ylabel = ylabel.cuda()
            # optimizer.zero_grad()
            # batch_output = []







    return 

### load data






## train per protein batchloss
#load data per protein 
TRAIN_yeast = dataset_perprotein(ppipath,esmfolder,svfolder)
prnames = TRAIN_yeast.prname
n2f = TRAIN_yeast.n2f
ddpos_neg = TRAIN_yeast.ddpos_neg
print('ddpos_neg:',ddpos_neg)

def train_sim_perprotine(NUMBER_EPOCHS,BATCH_SIZE,MODEL_SAVE,SAVEPATH,loadpath,New=True):
    print('SAVEPATH',SAVEPATH,flush = True)
    print('loadpath',loadpath,flush = True)
    net  = Featuremodel(1280,200,1280)
    print('Loading model....',flush = True)
    print('time=',time.asctime( time.localtime(time.time()) ), flush=True)
    net.train()
    # criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
    print('lr=1e-3',flush = True)
    start_epoch =-1
    print('dataloader time=',time.asctime( time.localtime(time.time()) ), flush=True)
    Trainloader = Data.DataLoader(dataset=TRAIN_yeast,
                            shuffle=False,
                            batch_size = BATCH_SIZE
                            )
##############
    def batchloss(n2f,ddpos_neg,prs):
        cos_sim_pos =[]
        cos_sim_neg =[]

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        for pr in prs:
            f0 = n2f[pr]
            f0 = torch.as_tensor(n2f[pr])
            try:
                pos_p = ddpos_neg['1'][pr]
                neg_p = ddpos_neg['0'][pr]
                print('len pos={}'.format(len(pos_p)))
                print('len neg={}'.format(len(neg_p)))
            except:
                continue
            
            for pp in pos_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_pos.append(output.item())

            for pp in neg_p:
                f1 = torch.as_tensor(n2f[pp])
                # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                output = cos(f0,f1)
                cos_sim_neg.append(output.item())
        try:
            posloss = 1 -  (sum(cos_sim_pos) / len(cos_sim_pos))
        except:
            # print('posloss =0')
            posloss =torch.tensor(0.0,requires_grad = True)
        try:
            negloss = sum(cos_sim_neg) / len(cos_sim_neg)
        except:
            # print('posloss =0')
            negloss = torch.tensor(0.0,requires_grad = True)
        totalloss = posloss + negloss
        # print('')
        return totalloss
## neg by rand (neg50-100>pos1)skip interactions with low confident 
# eval by protein with rank cos sim and 
# split proteins 

    for i,data in enumerate(Trainloader,0):
        # print('BATCH_SIZE=',BATCH_SIZE)
        # print('data=',data)
        for pr in data[0]:
            # print('prname=',pr)
            orif = torch.as_tensor(n2f[pr])
            newf = net(orif)
            n2f[pr] = newf
        # print('data[0]=',data[0])
        bloss = batchloss(n2f,ddpos_neg,data[0])
        print('bloss:',bloss)

        bloss.backward()
        optimizer.step()  

        if i>5:
            break
        
##############

    # testloader = Data.DataLoader(dataset=VAL_yeast,
    #                         shuffle=True,
    #                         batch_size=1
    #                     )
    # print('//time=',time.asctime( time.localtime(time.time()) ), flush=True)
    # print('Train $ Test initial finish : ',flush=True)
    # # if New == False:
    # #     print('loading check point...',flush = True)
    # #     checkpoint = torch.load(loadpath)  # 加载断点
    # #     net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
    # #     optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    # #     start_epoch = checkpoint['epoch'] 
    # #     print('start epoch ==',start_epoch,flush= True)


    # for epoch in range(start_epoch+1,NUMBER_EPOCHS+1):
    #     print('epoch',epoch,'start.',flush=True)
        
    #     print('time=',time.asctime(time.localtime(time.time())), flush=True)
    #     # torch.cuda.empty_cache()
    #     for i,data in enumerate(Trainloader,0):
    #         print('BATCH_SIZE=',BATCH_SIZE)
    #         print('data=',data)
            # p1 = data[0][0]
            # p2 = data[0][1]
            # print('p1=',p1)
            # print('p2=',p2)

### train
# train_sim_pairwise(20,4,0,'path','loap')
train_sim_perprotine(20,8,0,0,0,True)

