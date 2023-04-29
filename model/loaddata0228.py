## load from fp to data loader
import pandas
import torch
import glob
import numpy as np
import pandas as pd
import json
import random
import os
import time

class Loaddata():
    def __init__(self,highppipath,lowppipath,esmfolder,svfolder,negativefold,name) :
        self.esmfolder = esmfolder
        self.svfolder = svfolder
        self.name = name
        allinfo_file = f'{svfolder}{name}_allinfo.pkl'
        allpairs_file = f'{svfolder}{name}_allpairs.pkl'
        allpairs_csvfile = f'{svfolder}{name}_allpairs.csv'
        name2index_file = f'{svfolder}{name}_name2index.pkl'
        index2name_file = f'{svfolder}{name}_index2name.pkl'

        

        # allpairs_tsvfile = f'{svfolder}{name}_allpairs.csv'


        self.hppipath = highppipath 
        self.lppipath = lowppipath  
        self.negativefold=negativefold
        # print('> loading esm',flush=True)
        if not os.path.exists(allinfo_file):
            allinfo,index2name,name2index = self.loadesmfolder()
            print('finished loading esm',flush=True)

            self.allinfo = allinfo
            self.index2name=index2name
            self.name2index=name2index
            print('sv...',flush=True)
            allinfo.to_pickle(allinfo_file)
            with open(name2index_file,'w')as f:
                json.dump(name2index,f)
            with open(index2name_file,'w')as f:
                json.dump(index2name,f)
            print('finished allinfo',flush=True)
        else:
            self.allinfo = pd.read_pickle(allinfo_file)
            with open(name2index_file,'r')as f:
                name2index = json.load(f)
            with open(index2name_file,'r')as f:
                index2name = json.load(f)
            self.index2name=index2name
            self.name2index=name2index



        print('> loading pairs',flush=True)
        allpairs = self.loadallpairs()
        allpairs.to_csv(allpairs_csvfile)
        self.allpairs = allpairs
        print('allpairs top5\n',allpairs[:5])
        print('empty line count?=\n',allpairs[allpairs.isnull().T.any()])
        allpairs.to_pickle(allpairs_file)
        print('>>>finished allpairs\n',flush=True)

        # self.notnegpairs = notnegpairs

    def loadesmfolder(self):
        name2index = {}
        index2name=[]
        dd={}
        id =0
        # for i in range(0,200):
        #     name='name_'+str(id)
        #     fff = torch.rand(1280)
        #     dd[name] = fff
        #     name2index[name] = id
        #     index2name.append(name)
        #     id+=1
        

        for filename in glob.glob(self.esmfolder+'*'):
            name = filename.replace(self.esmfolder,'')
            name = name.replace('.pt','')
            fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
            # fff = torch.random(1,20)
            dd[name]=fff
            name2index[name] = id
            index2name.append(name)
            id+=1
            if len(dd.keys()) %1000 == 0:
                print('esm protein..\t',len(dd.keys()))

        allinfo = pd.DataFrame(dd.items(),columns=['name', 'feature'])
        

        allinfo = allinfo.reset_index()
        
        print('>>>allinfo\n',allinfo[:3])
        return allinfo,index2name,name2index

    def loadallpairs(self):
        
        allpairs_csvfile = f'{self.svfolder}{self.name}_allpairs.csv'


        if os.path.exists(allpairs_csvfile):
            allpairs = pd.read_csv(allpairs_file)
            return allpairs


        # allpairs.to_pickle(allpairs_file)
        pospairs_file = f'{self.svfolder}{self.name}_pospairs.csv'
        pospairs_tmpfile = f'{self.svfolder}{self.name}_tmppospairs.csv'

        if not os.path.exists(pospairs_file):
            pospairf = open(pospairs_tmpfile,'w+')
            print('{},{},{}'.format('p1','p2','label'),file = pospairf,flush=True)

            # pairs = pd.DataFrame(columns=['p1','p2','label'])
            print('empty df pairs',flush=True)
            with open(self.hppipath,'r') as f:
                print('hppipath pairs opened',flush=True)
                # label = torch.tensor([1])
                label = 1
                cc = 0
                print('cc = ',cc,flush=True)

                for line in f:
                    line = line.strip('\n')
                    p1,p2,_ = line.split('\t')
                    # print('p1=',p1)
                    p1index = self.name2index[p1]
                    p2index = self.name2index[p2]
                    # print('name2index = ',len(self.name2index.keys()),flush=True)
                    # print('p1index = ',p1index,flush=True)

                    # p2index = random.randint(0,60)                
                    # p1index = random.randint(0,60)
            
                    if p1index < p2index:
                        print('{},{},{}'.format(p1index,p2index,label),file = pospairf,flush=True)
                        # pairs = pairs.append(pd.DataFrame({'p1':[qp1index],'p2':[p2index],'label':[label]}))
                    else:
                        print('{},{},{}'.format(p2index,p1index,label),file = pospairf,flush=True)
                        # pairs = pairs.append(pd.DataFrame({'p1':[p2index],'p2':[p1index],'label':[label]}))
                    cc +=1
                    if cc % 2407 == 0: #240700/2400 = 100%
                        tt = time.asctime()
                        print('cc time=',tt)
                        print('pairs ..\t',cc,cc/2407*100,"%",flush=True)
            pospairf.close()
            os.rename(pospairs_tmpfile,pospairs_file)
            print('!!!!finished pospairs',flush=True)
            pospairs = pd.read_csv(pospairs_file)
            print('pospairs file read by pd====>\n',pospairs[:10])
            # pospairs.to_pickle(pospairs_file)
        else:
            pospairs = pd.read_csv(pospairs_file)
            print('else :pospairs file read by pd====>\n',pospairs[:10])

            # pairs = pd.read_pickle(pospairs_file)
            # pairs = pairs.astype(int)

        notnegpairs_file = f'{self.svfolder}{self.name}_notnegpairs.csv'
        notnegpairs_tmpfile = f'{self.svfolder}{self.name}_tmpnotnegpairs.csv'
        if not os.path.exists(notnegpairs_file):
            notnegpairsf = open(notnegpairs_tmpfile,'w+')
            print('{},{},{}'.format('p1','p2','label'),file = notnegpairsf,flush=True)
            # notnegpairs =pd.DataFrame(columns=['p1','p2','label'])
            with open(self.lppipath,'r') as f:
                # label = torch.tensor([-1])
                cn =0
                label = 2
                for line in f:
                    line = line.strip('\n')
                    p1,p2,_ = line.split('\t')
                    p1index = self.name2index[p1]
                    p2index = self.name2index[p2]
                    # p1index = random.randint(0,60)                
                    # p2index = random.randint(20,100)

                    if p1index < p2index:
                        print('{},{},{}'.format(p1index,p2index,label),file = notnegpairsf,flush=True)

                        # notnegpairs = notnegpairs.append(pd.DataFrame({'p1':[p1index],'p2':[p2index],'label':[label]}))
                    else:
                        print('{},{},{}'.format(p2index,p1index,label),file = notnegpairsf,flush=True)

                        # notnegpairs = notnegpairs.append(pd.DataFrame({'p1':[p2index],'p2':[p1index],'label':[label]}))
                    cn+=1

                    #1747820
                    if cn % 17478 == 0:
                        tn = time.asctime()
                        print('cn time=',tn)
                        print('notnegpairs ..\t',int(cn/1747820*100),"%",flush=True)

            notnegpairsf.close()
            os.rename(notnegpairs_tmpfile,notnegpairs_file)
            print('!!!!finished notnegpairs',flush=True)

            notnegpairs = pd.read_csv(notnegpairs_file)
            # notnegpairs_file = f'{svfolder}{name}_tmpnotnegpairs.pkl'
            # notnegpairs.to_pickle(notnegpairs_file)
        else:
            notnegpairs = pd.read_csv(notnegpairs_file)
            print('load notnegtives')
            # notnegpairs = pd.read_pickle(notnegpairs_file)

        negpairs_file = f'{self.svfolder}{self.name}_negpairs.csv'

        if not os.path.exists(negpairs_file):

            totalrange = len(self.index2name)
            def randomgene(notset:set,rangenumber,totalnumber):
                num =[]
                for i in range(0,totalnumber):
                    x = random.randint(0,rangenumber)
                    if x not in notset:
                        num.append(x)
                return num
            
            # pairs = pospairs
            negpairs_file = f'{self.svfolder}{self.name}_negpairs.csv'
            negpairs_tmpfile = f'{self.svfolder}{self.name}_tmpnegpairs.csv'
            negpairsf = open(negpairs_tmpfile,'w+')
            
            print('{},{},{}'.format('p1','p2','label'),file = negpairsf,flush=True)
            
            for protein in range (0,totalrange):
                # protein = 60
                # print('protein=',protein)
                if int(protein) % 1000 == 0:
                    print('protein negative generate ..\t',protein)

                posdf = pospairs.query('p1==@protein | p2==@protein')
                posindex = set(posdf['p2'])|set(posdf['p1'])
                # print('posindex==',posindex,flush=True)
                posnumber = len(posindex)
                if posnumber ==0:
                    continue
                try:
                    posindex.remove(int(protein))
                except:
                    print('len(posindex)=',posnumber)
                    print('prindex',protein)
                    print('prname',self.index2name[protein],flush= True)

                notnegdf = notnegpairs.query('p1==@protein & label==2')
                not_negindex = set(notnegdf['p2'])|set(notnegdf['p1'])
                # if len(not_negindex) ==0:
                    
                # else:  
                #     not_negindex.remove(protein)
                # # print('notngindex==',not_negindex)
                try:
                    not_negindex.remove(protein)
                except:
                    pass


                not_negindex =  posindex|not_negindex
                negindexlist = randomgene(not_negindex,totalrange,posnumber*self.negativefold)
                for p2ii in negindexlist:
                    print("{},{},{}".format(int(protein),int(p2ii),0),file = negpairsf, flush=True )

                
            
            negpairsf.close()
            os.rename(negpairs_tmpfile,negpairs_file)
            negpairs = pd.read_csv(negpairs_file)

            print('!!!!finished negpairsf',flush=True)
        else:
            negpairs = pd.read_csv(negpairs_file)
            print('load negatives')


     
        print('pos=',pospairs[:5])
        print('neg=',negpairs[:5])

        pairs = pd.concat([negpairs,pospairs])
        print('pairs=contact=',pairs[:5],flush=True)
        # pairs1 = pd.concat([negpairs,pospairs],axis = 0)
        # print('pairs=contact00=',pairs[:5])

        return pairs



