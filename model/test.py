# import glob
# def getESMembedding(esmfolderpath):
#     feature=[]
#     rnames = []
#     pp = esmfolderpath+'*'
#     for filename in glob.glob(pp):
#         name = filename.replace(esmfolderpath,'')
#         name = name.replace('.pt','')
#         rnames.append(name)
#         # fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
#         # feature.append(fff)
#     return feature,rnames

# esm1folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'
# all_data,all_name1 = getESMembedding(esm1folder)
# # print('len(all_)  1 =',len(all_name1))
# # esm2folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
# # all_data,all_name2 = getESMembedding(esm2folder)
# # print('len(all_)  2 =',len(all_name2))

# # dis = set(all_name2).difference(set(all_name1))
# # print(dis)




# #####################
# #### Read HDF5 ######
# #####################
# import h5py
# filename = "/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/per-protein.h5"

# with h5py.File(filename, "r") as f:
#     # Print all root level object names (aka keys) 
#     # these can be group or dataset names 
#     # print("Keys: %s" % f.keys())

#     sgd = set(all_name1).intersection(set(f.keys()))
#     print(sgd)
#     print('number= ',len(sgd))
#     sgd = list(sgd)
    


    # get first object name/key; may or may NOT be a group
    # a_group_key = list(f.keys())[0]

    # get the object type for a_group_key: usually group or dataset
    # print(type(f[a_group_key])) 

    # If a_group_key is a group name, 
    # this gets the object names in the group and returns as a list
    # data = list(f[a_group_key])

    # # If a_group_key is a dataset name, 
    # # this gets the dataset values and returns as a list
    # data = list(f[a_group_key])
    # # preferred methods to get dataset values:
    # ds_obj = f[a_group_key]      # returns as a h5py dataset object
    # ds_arr = f[a_group_key][()]  # returns as a numpy array

# import random
# def random_neg(notneg_pairs,pr,prs,kk):
#         negp = []
#         while len(negp) < kk:
            
#             lack = kk-len(negp)
            
#             print('lack={},kk={},len(prs)={}'.format(lack,kk,len(prs)))
#             currprlist = random.sample(prs,lack)
#             print('currprlist=',currprlist)
#             for curpr in currprlist:
#                 if curpr in negp:
#                     continue
#                 if curpr<pr and (curpr,pr) not in notneg_pairs:
#                     negp.append(curpr)
#                 elif curpr>pr and (pr,curpr) not in notneg_pairs:
#                     negp.append(curpr)
#         return negp
# notneg_pairs = [(1,2),(2,7),(1,3),(2,5),(1,4),(3,7)]
# prs = [1,2,7,3,4,5]
# pr = 5
# print(random_neg(notneg_pairs,pr,prs,4))
# import pandas

# from simdataset import dataset_prosplit

# high_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.highconf.links.txt'
# low_ppipath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.lowconf.links.txt'
# esmfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_4932/'

# svfolder='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/'

# TRAIN_yeast = dataset_prosplit(allinfo='',allpairs='',highppipath=high_ppipath,\
#                                 lowppipath=low_ppipath,esmfolder=esmfolder,\
#                                 svfolder=svfolder,datalabel='train')
# allinfo = TRAIN_yeast.allinfo
# allpairs = TRAIN_yeast.allpairs
# notneg_pairs = TRAIN_yeast.notneg

# n2f = TRAIN_yeast.n2f

# trainposdd = TRAIN_yeast.posdd

# import pandas as pd
# import torch as th
# from torch_utils import FastTensorDataLoader
# filepath ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/train_data.pkl'
# termfp ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/terms.pkl'


# def load_data(data_root, ont, terms_file):
#     terms_df = pd.read_pickle(terms_file)
#     terms = terms_df['gos'].values.flatten()
#     terms_dict = {v: i for i, v in enumerate(terms)}
#     print('Terms', len(terms))
    
#     # train_df = pd.read_pickle(f'{data_root}/{ont}/train_data.pkl')
#     train_df = pd.read_pickle(filepath)
#     # valid_df = pd.read_pickle(f'{data_root}/{ont}/valid_data.pkl')
#     # test_df = pd.read_pickle(f'{data_root}/{ont}/test_data.pkl')
#     print('train_df=\n',train_df.dtypes)
#     print('train_df=\n',train_df.info)

#     train_data = get_data(train_df, terms_dict)
#     # valid_data = get_data(valid_df, terms_dict)
#     # test_data = get_data(test_df, terms_dict)

#     return terms_dict, train_data

# def get_data(df, terms_dict):
#     data = th.zeros((len(df), 2560), dtype=th.float32)
#     labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
#     for i, row in enumerate(df.itertuples()):
#         data[i, :] = th.FloatTensor(row.esm2)
#         for go_id in row.prop_annotations: # prop_annotations for full model
#             if go_id in terms_dict:
#                 g_id = terms_dict[go_id]
#                 labels[i, g_id] = 1
#     return data, labels
# batch_size = 8
# terms_dict, train_data =load_data('','',termfp)
# train_features, train_labels = train_data
# # print('train_feature=\n',train_features)
# train_loader = FastTensorDataLoader(
# #         *train_data, batch_size=batch_size, shuffle=True)

# import pandas as pd
# import chardet
# # import pandas as pd
# svfolder ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/'
# notnegpairs_file = f'{svfolder}4932_notnegpairs.csv'
# print('>>notnegpairs_file||',notnegpairs_file,"\n",type(notnegpairs_file))


# # notnegpairs_file = f'{svfolder}4932_notnegpairs.csv'
# # pospairs_file = f'{svfolder}4932_pospairs.csv'
# # negpairs_file = f'{self.svfolder}4932_negpairs.csv'
# # with open(notnegpairs_file,'rb')as f:
# #     # result = chardet.detect(f.readline()) 
# #     # print(result)

# notnegpairs = pd.read_csv(notnegpairs_file, encoding = 'Windows-1254')
# print(notnegpairs[:3])



# import torch
# import pandas as pd

# vector x vector
# tensor1 = torch.randn(3)
# tensor2 = torch.randn(3)
# print('tensor1=',tensor1.size())
# print('tensor2=',tensor2.size())
# torch.matmul(tensor1, tensor2).size()
# print('a = ',torch.matmul(tensor1, tensor2))
# print('a = ',torch.matmul(tensor1, tensor2).size())
# matrix x vector
# tensor1 = torch.randn(3, 4)
# tensor2 = torch.randn(4)
# torch.matmul(tensor1, tensor2).size()
# batched matrix x broadcasted vector
# import pandas as pd
# # import torch
# ii =[10,2,5,4,9,6,7,8,3,1]
# df = pd.DataFrame({
#     "index":ii,
#     "nn":   [0,0,0,0,0,0,0,0,0,0],
#     "nn1":  [0,0,0,0,0,0,0,0,0,0]
# })
# print(df)
# tensor0 =  torch.tensor([[2.],
#         [4],
#         [5.],
#         [7.],
#         [2.]])
# # tensor0 =  torch.tensor([[2.]])
# tensor1 =  torch.tensor([[11],
#         [11],
#         [4],
#         [4],
#         [9.]])
# x1 = torch.tensor([[1,1,1,1,2,3,4,4,5],
# [1,1,1,1,2,3,4,4,5],
# [1,1,1,1,2,3,4,4,5],
# [1,1,1,1,2,3,4,4,5],
# [1,1,1,1,2,3,4,4,5],
# ])
# print('x1 size={}'.format(x1.size()))
# batchsize = 4
# number_out = 3
# x2 =torch.rand([batchsize,number_out,12])
# print('x2=\n{}'.format(x2))
# print('x2 size={}'.format(x2.size()))
# out1 = x2[-1]
# print('out1=\n{}'.format(out1))
# pairv = torch.rand([1,12])
# print('pairv={},|{}'.format(pairv,pairv.size()))
# currlis=[]
# for vv in out1:
#     print('vv={},|{}'.format(vv,vv.size()))
    
#     loss = torch.dot(vv.t(),pairv.squeeze())
#     currlis.append(loss)
#     print('loss ={}'.format(loss))
# loss = 0

# for x in currlis:
#     loss += batchsize/x
# print('loss={}'.format(loss))
# def add(a,b):
#     return a+b
# def minus(a,b):
#     return a-b
# cal = add
# print(cal(2,3))
# x0index = torch.squeeze(tensor0).tolist()

# print('x0index={}'.format(x0index))
# def aa(df):
#     for i in range(0,len(x0index)):

#         tmp0 = x1[i].detach().numpy().tolist()
#         mask = df['index']==x0index[i]
#         df.loc[mask,'nn1'] = pd.Series([tmp0],index = df.index[mask])
#         # print('df =\n{}'.format(df))
#     return df
# df1 = aa(df)
# print('df =\n{}'.format(df))
# print('df1 =\n{}'.format(df1))


# for i in range(0,len(x0index)):
#     tmp0 = x0[i].detach().numpy().tolist()
#     tmp1 = x1[i].detach().numpy().tolist()
    
#     mask = df['index'] == x0index[i]
#     mask1 = df['index'] == x1index[i]

#     df.loc[mask,'feature'] = pd.Series([tmp0], index=df.index[mask])
#     df.loc[mask1,'feature'] = pd.Series([tmp1], index=df.index[mask1])
    # print('assgin finished!')



# print('tensor1=',tensor1,tensor1.size())
# # print(tensor1.item())
# index = df['index']
# print('index={},|{}'.format('1',type(index)))
# index= index.tolist()
# print('index={},|{}'.format('1',type(index)))

# x0index = torch.squeeze(tensor0)
# x1index = torch.squeeze(tensor1)
# # x0index = tensor1

# # print('x0index1=',x0index)
# x0index = x0index.tolist()
# x1index = x1index.tolist()
# print('x0index11=',x0index)
# print('x1index11=',x1index)
# print('float={}'.format(type(x0index)))

# # f1 = df.loc[df['index'].isin(x0index)]
# # # f1 = df.loc[df['index'].isin(x0index)].nn.values.tolist()
# # print('f1=\n',f1)
# newx0index, newx1index = [],[]
# x0c,x1c = [],[]
# # x0index = 5.0
# print('float={}'.format(type(x0index)))

# if isinstance(x0index, float):
#     print('float={}'.format(type(x0index)))
#     x0index = [x0index]
#     print('======',x0index)
#     # print()
#     ff0 = df.loc[df['index']==x0index,'nn1'].values
#     # ff1 = df.loc[df['index']==x1index[i],'nn1'].values
#     print('ff0={}'.format(ff0))
#     print('ff1={}'.format(ff1))

# else:
#     for i in range(0,len(x0index)):
#         if x0index[i] in df['index'] and x1index[i] in df['index']:
            
#             ff0 = df.loc[df['index']==x0index[i],'nn1'].values
#             ff1 = df.loc[df['index']==x1index[i],'nn1'].values
#             print('ff0={}'.format(ff0))
#             print('ff1={}'.format(ff1))
            

#             x0c.append(torch.tensor(ff0))
#             x1c.append(torch.tensor(ff1))

#         # newx0index.append(x0index[i])
#         # newx1index.append(x1index[i])

# x0 = torch.stack(x0c)
# x1 = torch.stack(x1c)

# print('xstack0.=',type(x0),x0.size())
# print('xstack1.=',type(x1),x1.size())

# f0 = df.at[df['index']==newx0index,'feature']
# print('f0=\n',f0)


# f1 = df.iat[df['index']== newx1index]
# print('f1=\n',f1)

# f1 = df.loc[df['index'].isin(newx1index)]
# print('f1=\n',f1)


# tensor2 = torch.randn(5, 12)
# print('tensor1=',tensor1.size())
# print('tensor2=',tensor2.size())

# print('tensor2=',tensor2,'\n',tensor2.size())

# def bdot(a, b):
#     B = a.shape[0]
#     S = a.shape[1]
#     print('b={},s={}'.format(B,S))
#     return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

# a = bdot(tensor1,tensor2)
# # a = torch.matmul(tensor1, tensor2)
# print('a=',a.size())
# print(a)



# # # batched matrix x batched matrix
# # tensor1 = torch.randn(10, 3, 4)
# # tensor2 = torch.randn(10, 4, 5)
# # torch.matmul(tensor1, tensor2).size()
# # # batched matrix x broadcasted matrix
# # tensor1 = torch.randn(10, 3, 4)
# # tensor2 = torch.randn(4, 5)
# # torch.matmul(tensor1, tensor2).size()


# import torch
# tensor1 = torch.randn([2,64])
# print(tensor1[1],tensor1[1].size())
# xx =tensor1[1]
# print('xx size={}'.format(xx.size()))

# number =8
# xx = xx.reshape([number,int(xx.shape[0]/number)])
# print('xx size={}'.format(xx.size()))

# # tensor2 = torch.randn([64])
# a ,b,c =1,2,3
# kk = a,b
# print(kk)
# a = [1,2,3,4,5,6]
# b = [1,2,3,4,5,6]
# k = zip(a,b)
# kk = #zip 拼接

# for pair in kk :
#     print(pair)
# print('tensor1={},tensor2={}'.format(tensor1.size(),tensor2.size()))

# newf0 = torch.cat((tensor1,tensor2),0)
# newf1= torch.stack((tensor1,tensor2),1)
# print('newf0={},newf1={}'.format(newf0.size(),newf1.size()))

# a =[]
# if a:
#     print('dd')
# else:
#     print('right')

# import torch

# a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059]])

# b = torch.tensor([[0.0,0.0], [0.0,0.0],[0.0,0.0]])

# # ['dot','Euclidean distance','Manhattan metric','Chebychev metric']

# eu = torch.cdist(a, b, p=2,compute_model='use_mm_for_euclid_dist')
# # eu = torch(a, b, p=2)
# print(eu)

# import torch
# pdist = torch.nn.PairwiseDistance(p=1)
# # p=1  ManhattanDistance

# edist = torch.nn.PairwiseDistance(p=2)
# # p=2 euclidean

# # x0 = torch.tensor([0., 1., 2., 3., 4., 5.],[0., 1., 2., 3., 4., 5.],[0., 1., 2., 3., 4., 5.])
# x0 = torch.randn(3,5)
# print('0=',x0)

# x1 = x0 * 0.75
# print('x1=',x1)

# x4 = x0*0.25
# summ=torch.sum((x1[0]-x0[0])**2)
# print('1=',summ)
# summ = torch.sqrt(summ)
# print('2=',summ)

    
# manhaval = pdist(x0, x1)
# euclival = edist(x0, x1)

# print('manhattan={}|{}'.format(manhaval,torch.sum(x4[0])))
# print('euclival={}|{}'.format(euclival,summ))


import torch
import pandas as pd
import numpy as np
from torch_utils import FastTensorDataLoader
from torch import nn

allinfo_file ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932_allinfo.pkl'
allinfo = pd.read_pickle(allinfo_file)
# print(allinfo[:5])
# all_data = allinfo.feature,allinfo.index
# feature = allinfo['feature'].values
# print('type={}'.format(type(feature)))

a = allinfo['feature'].values
a = np.vstack(a)
ff = torch.from_numpy(a).float()
# print('ff=',ff[5],type(ff[0]),len(ff[0]),type(ff))

a = allinfo['index'].values
a = np.vstack(a).astype(np.double)
ii = torch.from_numpy(a).float()
# .type('torch.DoubleTensor')
# print('ii=',ii[5],type(ii[0]),len(ii[0]),type(ii))


all_data = ff,ii
all_loader = FastTensorDataLoader(*all_data,batch_size=4,shuffle = False)
class Featuredotmodel(nn.Module):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        output = self.Sigmoid(dotproduct)
        return output

    def predict(self,x:torch.Tensor):
        # x0 here is the esm2 embedding 
        #return the 
        x0 = self.linear1(x)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        return x0

class twoFeaturedotmodel(nn.Module):
    #twof with relu works
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.linear3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear4 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        print('relu as activation between layers',flush=True)
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        x0,x1 = x.chunk(2,axis=1)
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
        
        # print('dotproduct',dotproduct.size(),flush=True)
        output = self.Sigmoid(dotproduct)
        return output

    def predict(self,x:torch.Tensor)-> torch.Tensor:

        
        x0 = self.linear1(x)
        # x0 = self.Sigmoid(x0)
        x0 = self.relu(x0)
        x0 = self.linear2(x0)

        x1 = self.linear3(x)
        # x1 = self.Sigmoid(x1)
        x1 = self.relu(x1)
        x1 = self.linear4(x1)
        return x0,x1


# modelpath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_50_onef_dot_64.th'
modelpath='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/model_esm2_30_twof_dot_64.th'
net = twoFeaturedotmodel(in_dim=1280,hidden_dim=100,out_dim=1280)
net.load_state_dict(torch.load(modelpath))
# print('ffff={}'.format(feature))
dd =[]
with torch.no_grad():
    for batch_features,bathch_indexs in all_loader:
        batch_features1,batch_features2 = net.predict(batch_features)
        print('typ=',type(batch_features1))
        print('size=',batch_features1.size())
        bathch_indexs = torch.squeeze(bathch_indexs)
        print ('bathch_indexs:\n',bathch_indexs.tolist())
        for i in range(0,len(bathch_indexs)) :
            ind = bathch_indexs[i]
            ind = int(ind)
            print('type ={},|{}'.format(type(ind),ind))
            # dd[int(ind)] = batch_features[i]
            dd.append([int(ind),batch_features1[i].tolist(),batch_features2[i].tolist()])

        predallinfo = pd.DataFrame(dd ,columns=['index','predf1','predf2'])
        print(predallinfo)
        break
        

