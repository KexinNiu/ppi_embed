import numpy as np
from datasketch import WeightedMinHashGenerator
from datasketch import MinHashLSHForest
from tqdm import tqdm
import os
import json
import glob
import torch
import random
print('...loading ',flush=True)

# import glob
# def getESMembedding(esmfolderpath):
#     ##ori version
#     feature=[]
#     rnames = []
#     pp = esmfolderpath+'*'
#     for filename in glob.glob(pp):
#         name = filename.replace(esmfolderpath,'')
#         name = name.replace('.pt','')
#         rnames.append(name)
#         fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
#         feature.append(fff)
#     return feature,rnames

def getESMembedding(esmfolderpath): 
    ##从esm1 的路径获取名字 +esm2 的文件夹获取embedding
    orip = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'
    feature=[]
    rnames = []
    pp = orip+'*'
    for filename in glob.glob(pp):
        name = filename.replace(orip,'')
        name = name.replace('.pt','')
        rnames.append(name)
        filename = filename.replace(orip,esmfolderpath)
        fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
        feature.append(fff)
    return feature,rnames



def getdgpembedding(esmfolderpath):
    ##从esm1 的路径获取名字 +dgp 的文件夹获取embedding
    orip = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'

    feature=[]
    rnames = []
    pp = orip+'*'
    for filename in glob.glob(pp):
        name = filename.replace(orip,'')
        name = name.replace('.pt','')
        rnames.append(name)
        filename = filename.replace(orip,esmfolderpath)
        fff = torch.load(filename).detach().numpy().tolist()
        feature.append(fff)
    return feature,rnames

import h5py

def getprot5embedding(h5fp):
    ##从esm1 的路径获取名字 +dgp 的文件夹获取embedding
    orip = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'
    feature=[]
    rnames = []
    pp = orip+'*'
    for filename in glob.glob(pp):
        name = filename.replace(orip,'')
        name = name.replace('.pt','')
        rnames.append(name)
    with h5py.File(h5fp, "r") as f:
        for prname in rnames:
            data = list(f[prname])
            feature.append(data)
    return feature,rnames
    



g2pp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_go2pr.json'
pr2gp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/sgd_pr2go.json'

#### Minhashlsh eval esm ####
def eval_esmembeding(embfolder,go2prp,pr2gop,svp,k=20,embname='esm'):
    outf = open(svp,'w')
    print('esmfolder=',embfolder,file=outf)
    print('topk={}'.format(k),file=outf)
    if embname =='esm':
        all_data,all_name = getESMembedding(embfolder)
    elif embname =='dgp':
        all_data,all_name = getdgpembedding(embfolder)
    elif embname =='prot5':
        all_data,all_name = getprot5embedding(embfolder)
    print('all_data.len',len(all_data))

    all_data= np.asarray(all_data)
    print('all_data.shape',all_data.shape)

    print('total data number={}'.format(len(all_data)),file=outf)

    mg = WeightedMinHashGenerator(all_data.shape[1],sample_size=128)
    forest = MinHashLSHForest()
    for index, value in tqdm(enumerate(all_data)):
        m_hash = mg.minhash(value)
        forest.add(index, m_hash)
    forest.index()  # 重要！在此之后才可以使用查询功能
    ##load go annotation 
    with open(pr2gop,'r') as f:
        pr2godd = json.load(f)

    ###enumerate all pr check top 20
    

    allqueryinfo =[len(all_name),0,0,0,0,0]#total nointersection ratio >=0.8 ratio >=0.6 ratio >=0.5 ratio <=0.1
    randomquery = [len(all_name),0,0,0,0,0]
    print('....querying',file=outf)
    for idx in range(0,len(all_name)):
        name = all_name[idx]
        emb = all_data[idx]
        go = pr2godd[name]
        query = all_data[idx]
        result = forest.query(mg.minhash(query), k)
        
        randomresult = []
        for i in range(0,k):
            n = random.randrange(len(all_name))
            # if n==4981:
            #     print('!!!!!!')
            randomresult.append(n)
        # print('len(all_name):',len(all_name),max(randomresult))

        # print('randresult:',len(randomresult),max(randomresult))
        def evalresult(result,allqueryinfo,PRINT):
            cqcc =0
            for topidx in result:
                rego = pr2godd[all_name[topidx]]
                if len(set(rego).intersection(set(go)))!=0:
                    cqcc +=1
            ratio = cqcc/k
            if cqcc ==0:
                allqueryinfo[1] +=1
            elif ratio<=0.1:
                allqueryinfo[5] +=1
            elif ratio>=0.8:
                allqueryinfo[2] +=1
            elif ratio>=0.6:
                allqueryinfo[3] +=1
            elif ratio>=0.5:
                allqueryinfo[4] +=1
            if PRINT ==True:
                print('>{}/{},{}'.format(cqcc,k,cqcc/k),file=outf)
            return  allqueryinfo
        allqueryinfo = evalresult(result,allqueryinfo,PRINT = True)
        randomquery = evalresult(randomresult,randomquery,PRINT = False)
    print('#total\tnointersection\tratio >=0.8\tratio >=0.6\tratio >=0.5\tratio <=0.1',file = outf)
    print(allqueryinfo,file=outf)
    print('random query',file = outf)
    print(randomquery,file=outf)


# KK=50
# kklist=[20,50,100,300]
# esm1folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'
# esm2folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'

# for KK in kklist:
#     ## 2rd 
#     # TODO: 'def getESMembedding(esmfolderpath):'
#     print('....done esm1 loading ',flush=True)
#     svp='./esm1minlsh_top{}_out.txt'.format(KK)
#     eval_esmembeding(esm1folder,g2pp,pr2gp,svp,k=KK,embname='esm')
#     print('....done esm2 loading ',flush=True)
#     svp='./esm2minlsh_top{}_out.txt'.format(KK)
#     eval_esmembeding(esm2folder,g2pp,pr2gp,svp,k=KK,embname='esm')



# #####eval dgp##########
# KK=50
# kklist=[20,50,100,300]
# dgpfolder ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/dgp_sgd/'
# for KK in kklist:
#     ## 1 ori 
#     # TODO: 'def getESMembedding(esmfolderpath):'
#     print('....done dgp loading ',flush=True)
#     svp='./dgpminlsh_top{}_out.txt'.format(KK)
#     eval_esmembeding(dgpfolder,g2pp,pr2gp,svp,k=KK,embname='dgp')

# ########################
#####eval ProtT5##########
# KK=50
kklist=[20,50,100,300]
# dgpfolder ='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/dgp_sgd/'
h5filep = "/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/per-protein.h5"

for KK in kklist:
    ## 1 ori 
    # TODO: 'def getESMembedding(esmfolderpath):'
    print('....done dgp loading ',flush=True)
    svp='./prot5minlsh_top{}_out.txt'.format(KK)
    eval_esmembeding(h5filep,g2pp,pr2gp,svp,k=KK,embname='prot5')

########################



##########load esm1 embed########
# esm1folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm1_sgd/'
# # all_data,all_name = getESMembedding(esm1folder)
# print('....done esm1 loading ',flush=True)
# svp='./esm1minlsh_top{}_out.txt'.format(KK)
# eval_esmembeding(esm1folder,g2pp,pr2gp,svp,k=KK,embname='esm')

##########load esm1 embed########

##########load esm2 embed########
# esm2folder = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
# # all_data,all_name = getESMembedding(esm2folder)
# print('....done esm2 loading ',flush=True)
# svp='./esm2minlsh_top{}_out.txt'.format(KK)
# eval_esmembeding(esm2folder,g2pp,pr2gp,svp,k=KK,embname='esm')
##########load esm2 embed########




        

# query = all_data[ii]
# result = forest.query(mg.minhash(query), 20)

###############load data ###########

# ecolip ='/ibex/scratch/projects/c2014/kexin/ppiproject/esm/data/1by1_ecoli/'
# def getESMembedding(esmfolderpath):
#     feature=[]
#     fnames =[]
#     def findAllFile(dirname):
#         result = []
#         rnames = []
#         for maindir, subdir, file_name_list in os.walk(dirname):
#             for filename in file_name_list:
#                 apath = os.path.join(maindir, filename)
#                 result.append(apath)
#                 rnames.append(filename)
#         return result,rnames

#     result,rnames =findAllFile(esmfolderpath)
#     for i in range(0,len(result)):
#         embpath = result[i]
#         name = rnames[i]
#         # print(i,'--',embpath,name,flush=True)
        
#         currf = np.load(embpath,'r+')
#         # print('???',currf,flush=True)
#         feature.append(currf)
#         fnames.append(name)
   
    
#     return feature,fnames


# all_data,all_name = getESMembedding(ecolip)
# print('....done loading ',flush=True)

# # all_data = np.random.random([10000, 100])
# # query = np.random.random([100])
##########load model embed########
# svp = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/'
# svclass = svp + 'embed_classonto.json'
# with open(svclass,'r') as f:
#     outputs_classify =json.load(f)

# all_data =[]
# split =[0,]
# for key in outputs_classify.keys():
#     if len(outputs_classify[key]) >=50:
#         all_data += outputs_classify[key]
#         split.append(len(outputs_classify[key])+split[-1])
#         # print('>>>>',len(outputs_classify[key]),len(outputs_classify[key][0]))
#         # >>>> 98 60
# # print(all_data.shape)
# print('split',split)
# # split [51, 137, 433, 233, 56, 370, 466, 98, 54, 87, 82]
# ##########load model embed########
# ###############load data ###########
# ll = len(all_data)
# print('ll=',ll)
# # query = np.random.random([ll])
# # print('query=',query)
# all_data= np.asarray(all_data)
# print(all_data.shape)

# # print('all_data.shape[1] =[0] ',all_data.shape[1],all_data.shape[0])
# # query = np.random.random([all_data.shape[1]])

# mg = WeightedMinHashGenerator(all_data.shape[1],sample_size=128)
# forest = MinHashLSHForest()
# for index, value in tqdm(enumerate(all_data)):

#     m_hash = mg.minhash(value)
#     forest.add(index, m_hash)
   

# forest.index()  # 重要！在此之后才可以使用查询功能


# # split [51, 137, 433, 233, 56, 370, 466, 98, 54, 87, 82]
# split = [0, 51, 188, 621, 854, 910, 1280, 1746, 1844, 1898, 1985, 2067]
# qlist=[x-20 for x in split[1:]]
# for i in range (0,len(qlist)):
#     ii = qlist[i]
#     print('ii={}'.format(ii))
#     query = all_data[ii]
# # query =all_data[0]

#     result = forest.query(mg.minhash(query), 20)  # 选择top20
#     # print('result')

#     # print(result)
#     cc = 0
#     for re in result:
#         if re<= split[i+1] and re >= split[i]:
#             cc+=1
#     print('>{}/20,{}'.format(cc,cc/20))



# # # 开始验证
# # print(np.sum(np.power(all_data[result] - query, 2), axis=-1))  # 计算LSH的结果与query的结果的差距
# # total_data = np.concatenate((all_data, [query]))
# # sort = np.argsort(np.sum(np.power(total_data - query, 2), axis=-1))  # 线性查找真正的最接近的曲线
# # print(np.sum(np.power(total_data[sort[:20]] - query, 2), axis=-1))  # 计算最接近的曲线
# # print(np.sum(np.power(total_data[sort[-20:]] - query, 2), axis=-1))
