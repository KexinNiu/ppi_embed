import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from utils import Ontology
from dataset import loaddataset
import pickle
import json

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




batch_size = 1
fastarefdictp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_reffasta.json'
g2pp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_go2pr.json'
pr2gp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_pr2go.json'
# Train=True

testset = loaddataset(fastarefdictp,g2pp,pr2gp,Train=False)
names = testset.idx2prnamelist
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# classes = set(testset.classes)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(1,2000, 5)

        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.embed = nn.Embedding(4,1,2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(42000, 120)

        self.fc1 = nn.Linear(42000, 120)
        self.fc2 = nn.Linear(120, 60)

        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # print('0x = ',x.size())
        x = self.embed(x)
        # print('embed x = ',x.size())

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # print('flat embed x = ',x.size())

        x = self.fc1(x)
        # print('1x = ',x.size())

        x = F.relu(x)
        # print('relu x = ',x.size())
        x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x


net = Net()
PATH = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/test_batch512_net.pth'
net.load_state_dict(torch.load(PATH))
net.eval()
def eval_embed(                 ):
    return     
# outputs_classify={}
# allembed = []
# # testset.
# for i, data in enumerate(testloader, 0):
        
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
        
#         outputs = net(inputs.type(torch.LongTensor))
#         #outputs embedding 
#         # labels ontoset 
        
#         for j in range (0,len(labels)):
#             labelonto = labels[j]
#             seqembed = outputs[j]
#             # print('labelonto = ',len(labelonto),labelonto)
#             allembed.append(seqembed) 
#             seqembed = seqembed.detach().numpy().tolist()
#             try:
#                 outputs_classify[labelonto].append(seqembed)
#             except:
#                 outputs_classify[labelonto] =[seqembed]

#             # for ontoterm in labelonto:
#             #     try:
#             #         outputs_classify[ontoterm].append(seqembed)
#             #     except:
#             #         outputs_classify[ontoterm] =[seqembed]
svp = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/'
svclass = svp + 'embed_classonto.json'
# svembe = svp + 'allembed.pickle'
# print('sv....',flush=True)

# with open(svembe,'wb') as f:
#     pickle.dump(allembed, f)

# with open(svclass,'w') as f:
#     json.dump(outputs_classify,f)

# print('finished sv')

# with open(svclass,'r') as f:
#     outputs_classify =json.load(f)
# ll = []
# for key in outputs_classify.keys():
#     ll.append(len(outputs_classify[key]))
# print(ll)
'''
[51, 137, 13, 433, 12, 233, 31, 56, 370, 466, 98, 54, 1, 12, 87, 41, 26, 13, 18, 2, 22, 47, 82, 5, 4, 16, 31, 47, 1, 44, 45, 10, 17, 34, 2, 5, 4, 32, 2, 12, 7, 15, 35, 1, 37, 4, 13, 25, 13, 8, 5, 1, 12, 5, 17, 11, 1, 6, 19, 13, 26, 22, 5, 17, 1, 1, 2, 1, 5, 25, 8, 18, 9, 1, 2, 7, 8, 10, 1, 2, 4, 1, 3, 5, 1, 3, 4, 13, 6, 3, 2, 2, 8, 2, 4, 1, 4, 1, 4, 3, 4, 4, 4, 2, 1, 6, 1, 1, 1, 4, 1, 1, 2, 3, 2, 3, 1, 1, 5, 4, 1, 1, 3, 3, 2, 1, 3, 1, 2, 1, 2, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 2, 3, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1]
'''