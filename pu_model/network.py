import torch
import numpy
from torch import nn
from torch.nn import functional as F
class Featuredotmodel(nn.Module):
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        # self.__in_dim = in_dim
        # self.__out_dim = out_dim
        # self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        # self.Sigmoid = nn.Sigmoid()
        self.Sigmoid = nn.ReLU()
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x0,x1:torch.Tensor)-> torch.Tensor:

        # x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        # print('shape={}'.format(x0.shape),flush=True)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        return dotproduct

class twoFeaturedotmodel(nn.Module):
    #twof with relu works
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        # self.__in_dim = in_dim
        # self.__out_dim = out_dim
        # self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.linear3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear4 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        print('relu as activation between layers',flush=True)
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x0,x1:torch.Tensor)-> torch.Tensor:

        # x0,x1 = x.chunk(2,axis=1)
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
        
        return dotproduct


# def PUloss(output, batch_labels):
#     bb = 5
#     pos = output[batch_labels == 1]
#     neg = output[batch_labels != 1]
#     # print('pos\n{}'.format(pos))
#     # print('neg\n{}'.format(neg))
#     # # pos_loss = torch.mean(torch.log(pos))
#     # pos_loss= F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
#     # neg_loss = torch.mean(torch.log(1-torch.sigmoid(neg)))
#     # print('pos_loss----\t{}'.format(pos_loss))
#     # print('neg_loss----\t{}'.format(neg_loss))

#     if len(pos) ==0:
#         pos_loss = torch.tensor([10.0]).cuda()
#     else:
#         pos_loss = torch.mean(pos)
#     neg_loss = torch.mean(neg)
    
#     loss = pos_loss - neg_loss - bb
#     # if 3. in batch_labels:
#     #     print('{}\t{} p-n ==>{}'.format(pos_loss.item(),neg_loss,loss))

#     logsig = nn.LogSigmoid()
#     if loss >500:
#         loss =torch.tensor([500.0]).cuda()
#     elif loss<-500:
#         loss=-torch.tensor([-500.0]).cuda()
#     logloss = -logsig(loss)

#     # print('loss--{}'.format(logloss))

#     return logloss

# def PUloss(output, batch_labels):
#     bb = 5
#     pos = output[batch_labels == 1]
#     neg = output[batch_labels != 1]
#     # print('pos\n{}'.format(pos))
#     # print('neg\n{}'.format(neg))
#     pos_loss = torch.mean(torch.sigmoid(pos))
#     # pos_loss= F.binary_cross_entropy_with_logits(torch.sigmoid(pos), torch.ones_like(pos))
#     neg_loss = torch.mean(torch.log(1-torch.sigmoid(neg)))
#     # print('pos_loss----\t{}'.format(pos_loss))
#     # print('neg_loss----\t{}'.format(neg_loss))

#     # if len(pos) ==0:
#     #     pos_loss = torch.tensor([10.0]).cuda()
#     # else:
#     #     pos_loss = torch.mean(pos)
#     # neg_loss = torch.mean(neg)
    
#     loss = pos_loss - neg_loss - bb
#     # if 3. in batch_labels:
#     #     print('{}\t{} p-n ==>{}'.format(pos_loss.item(),neg_loss,loss))

#     logsig = nn.LogSigmoid()
#     # if loss >500:
#     #     loss =torch.tensor([500.0]).cuda()
#     # elif loss<-500:
#     #     loss=-torch.tensor([-500.0]).cuda()
#     logloss = -logsig(loss)

#     print('loss--{}'.format(logloss))

#     return logloss

def PUloss(output, batch_labels):
    

    pos = output[batch_labels == 1]
    neg = output[batch_labels != 1]
    ##binary:
    sig = torch.nn.Sigmoid()
    output = sig(output)
    # print('output-{}'.format(output[:3]))
    loss = F.binary_cross_entropy(output, batch_labels)
    loss2 = torch.mean(torch.sigmoid(pos)) - torch.mean(torch.log(1-torch.sigmoid(neg)))
    # print('batch_labels-{}'.format(batch_labels[:3]))
    # print('loss-{}'.format(loss))
    loss = loss +loss2

    return loss
    


#train loss
# train 只有 1/0
# val 有123
##binary
def PUloss(output, batch_labels):
    # pos1 not neg 2 unlabel3
    # print('len{}|{}'.format(len(output),len(batch_labels)),flush= True)

    pos = output[batch_labels == 1.0]
    # notneg = output[batch_labels == 2]
    # neg = output[batch_labels == 0.0]
    neg = output[batch_labels != 1.0]
    # print('len{}|{}|{}'.format(len(pos),len(notneg),len(neg)),flush= True)
    # print('len{}|{}|{}'.format(len(pos),0,len(neg)),flush= True)
    pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    # notloss = (torch.mean(pos)-torch.mean(notneg)) + (torch.mean(notneg)-torch.mean(neg))
    # print('pos_loss-{}+neg_loss-{}+notloss-{}'.format(pos_loss,neg_loss,notloss),flush=True)
    # print('pos_loss-{}+neg_loss-{}'.format(pos_loss,neg_loss),flush=True)
    
    pos_loss[torch.isnan(pos_loss)] = 0.0
    neg_loss[torch.isnan(neg_loss)] = 0.0
    # notloss[torch.isnan(notloss)] = 0.0

    # pos_loss = torch.nan_to_num(pos_loss, nan=0.0)
    # neg_loss = torch.nan_to_num(neg_loss, nan=0.0)
    # notloss = torch.nan_to_num(notloss, nan=0.0)
    # print('pos_loss-{}+neg_loss-{}+notloss-{}'.format(pos_loss,neg_loss,notloss),flush=True)
    # print('pos_loss-{}+neg_loss-{}'.format(pos_loss,neg_loss),flush=True)

    # loss = pos_loss+neg_loss+notloss
    loss = pos_loss+neg_loss
    return loss
    
##binary
# def binary_PUloss(output, batch_labels):
#     # pos1 not neg 2 unlabel3# train 只有 1/0 
#     print('len{}|{}'.format(len(output),len(batch_labels)),flush= True)

#     pos = output[batch_labels == 1.0]
#     neg = output[batch_labels != 1.0]
    
#     print('len{}|{}|{}'.format(len(pos),0,len(neg)),flush= True)
#     pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
#     neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    
#     print('pos_loss-{}+neg_loss-{}'.format(pos_loss,neg_loss),flush=True)
    
#     pos_loss[torch.isnan(pos_loss)] = 0.0
#     neg_loss[torch.isnan(neg_loss)] = 0.0
    
#     print('pos_loss-{}+neg_loss-{}'.format(pos_loss,neg_loss),flush=True)
#     loss = pos_loss+neg_loss
#     return loss

##tri
def tri_PUloss(output, batch_labels):
    # pos1 not neg 2 unlabel3
    # print('len{}|{}'.format(len(output),len(batch_labels)),flush= True)

    pos = output[batch_labels == 1.0]
    notneg = output[batch_labels == 2]
    neg = output[batch_labels == 3.0]
    
    # print('len{}|{}|{}'.format(len(pos),len(notneg),len(neg)),flush= True)
   
    pos_loss = 10 * torch.nn.functional.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
    # notloss = (torch.mean(pos)-torch.mean(notneg)+5) + (torch.mean(notneg)-torch.mean(neg)+5)
    notloss =  torch.mean(torch.mean(notneg) + torch.mean(neg))- torch.mean(pos) +5
    

    # print('pos_loss-{}+neg_loss-{}+notloss-{}'.format(pos_loss,neg_loss,notloss),flush=True)
    
    
    pos_loss[torch.isnan(pos_loss)] = 0.0
    neg_loss[torch.isnan(neg_loss)] = 0.0
    notloss[torch.isnan(notloss)] = 0.0

  
    # print('pos_loss-{}+neg_loss-{}+notloss-{}'.format(pos_loss,neg_loss,notloss),flush=True)


    loss = pos_loss+neg_loss+notloss
    
    return loss
 