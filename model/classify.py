import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
from utils import Ontology
from dataset import loaddataset

import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument('--batchsize', type=int)
argparse.add_argument('--epchonum',type=int)
params = argparse.parse_args()


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = params.batchsize
fastarefdictp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_reffasta.json'
g2pp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_go2pr.json'
pr2gp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_pr2go.json'
Train=True
trainset = loaddataset(fastarefdictp,g2pp,pr2gp,Train=True)
testset = loaddataset(fastarefdictp,g2pp,pr2gp,Train=False)


# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = set(trainset.classes)
# classes = ('plane', 'car', 'bird', 'cat',
        #    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print('classes:',classes)

# functions to show an image


# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# for datainfo in enumerate(trainloader):
#     print('batch_size =',datainfo)
    




# get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# print('images=',len(images))

# print('labels=',len(labels))
# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.nn as nn
import torch.nn.functional as F


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


import torch.optim as optim

# criterion = nn.CrossEntropyLoss()


onto = Ontology('/ibex/scratch/projects/c2014/kexin/ppiproject/onto/data/go.obo')

def loss_onto(output,targets):
    def jaccard(a,b):
        return len(set(a).intersection(set(b))) / len(set(a).union(set(b)))
    #target current batch all ontoset 
    fulltarget = []
    for ontoset in targets:
        fullset = onto.get_anchestors(ontoset)
        fulltarget.append(fullset)
    loss = 0
    thr = 0.7
    neglabel = torch.tensor([-1])
    poslabel = torch.tensor([1])
    
    # print('len(fulltarget),',len(fulltarget))
    for i in range(0,len(fulltarget)-1):
        for j in range(i,len(fulltarget)-1):
            # print(' {}={}'.format(i,j))
            set1 = fulltarget[i]
            set2 = fulltarget[j]
            jac_dis = jaccard(set1,set2)
            # print('jac_di=',jac_dis)

            cosemb = nn.CosineEmbeddingLoss(margin =0,reduction='none')
            # print('len=output=',len(output),outputs.size())
            # print('output[1]',output[1])
            # print('i=',i,'j=',j)
            emb1 = output[i]
            emb2 = output[j]
            
            emb1 = emb1.unsqueeze(0)
            emb2 = emb2.unsqueeze(0)
            # print('emb1 = ',emb1)
            if jac_dis< thr:
                # print('emb1=',emb1.size(),neglabel.size())
                cosloss = cosemb(emb1,emb2,neglabel)
                # print('cosloss=',cosloss)
                loss+= cosloss*2
            else:
                cosloss = cosemb(emb1,emb2,poslabel)
                # print('cosloss=',cosloss)
                loss+=cosloss   
    # print('_batch loss')

            # emb1 = output[i]
            # emb2 = output[j]
            # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            # cosdis = cos(emb1, emb2)


            # shared = len(set1.intersection(set2))
            # total = 0.5*(len(set1) + int(len(set2)))
            # if shared !=0:
            #     currloss= total/shared
            # else:
            #     pass
            # if currloss ==1





    # inputfull = onto.get_anchestors(input)
    # targetfull = onto.get_anchestors(target)
    # shared = len(inputfull.intersection(targetfull))
    # total = 0.5*(len(inputfull) + int(len(targetfull)))
    # if shared ==0:
    #     loss= total/shared
    return loss

criterion = loss_onto

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

EPOCH = params.epchonum
for epoch in range(EPOCH):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #input  seqs 
        #labels  onto set

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.type(torch.LongTensor))
        #outputs embedding 
        # labels ontoset 

        loss = criterion(outputs,labels)
        print('batchloss= ',loss,flush=True)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

PATH = './test_batch{}_net{}.pth'.format(batch_size,EPOCH)
torch.save(net.state_dict(), PATH)

# dataiter = iter(testloader)
# images, labels = next(dataiter)

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))