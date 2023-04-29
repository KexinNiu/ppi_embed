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
    def __init__(self,highppipath,lowppipath,esmfolder,svfolder,negativefold) :
        self.esmfolder = esmfolder
        
        allinfo_file = f'{svfolder}4932_allinfo.pkl'
        allpairs_file = f'{svfolder}4932_allpairs.pkl'
        name2index_file = f'{svfolder}4932_name2index.pkl'
        index2name_file = f'{svfolder}4932_index2name.pkl'

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
            allinfo = pd.read_pickle(allinfo_file)
            with open(name2index_file,'r')as f:
                name2index = json.load(f)
            with open(index2name_file,'r')as f:
                index2name = json.load(f)
            self.index2name=index2name
            self.name2index=name2index



        print('> loading pairs',flush=True)
        allpairs = self.loadallpairs()
        self.allpairs = allpairs
        allpairs = allpairs.astype(int)
        allpairs.to_pickle(allpairs_file)

        print('finished allpairs',flush=True)

        # self.notnegpairs = notnegpairs

    def loadesmfolder(self):
        name2index = {}
        index2name=[]
        dd={}
        id =0
    
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
        # print('>>>allinfo\n',allinfo[:5])

        allinfo = allinfo.reset_index()
        
        print('>>>allinfo\n',allinfo[:3])
        return allinfo,index2name,name2index

    def loadallpairs(self):
        # allpairs.to_pickle(allpairs_file)
        pospairs_file = f'{svfolder}4932_tmpposallpairs.pkl'
        if not os.path.exists(allpairs_file):

            pairs = pd.DataFrame(columns=['p1','p2','label'])
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
                    print('p1=',p1)
                    p1index = self.name2index[p1]
                    p2index = self.name2index[p2]
                    print('name2index = ',self.name2index,flush=True)
                    
                    print('p1index = ',p1index,flush=True)

                    # p2index = random.randint(0,60)                
                    # p1index = random.randint(0,60)
            
                    if p1index < p2index:
                        pairs = pairs.append(pd.DataFrame({'p1':[p1index],'p2':[p2index],'label':[label]}))
                    else:
                        pairs = pairs.append(pd.DataFrame({'p1':[p2index],'p2':[p1index],'label':[label]}))
                    cc +=1
                    if cc % 2400 == 0: #240700/2400 = 100%
                        tt = time.asctime()
                        print('cc time=',tt)
                        print('pairs ..\t',cc/2400*100,"%",flush=True)
            
            # ###pairs = pairs.reset_index()
            # print('[pais=\n',pairs[:6])
        
            # cc = pairs.query('p1==20 & label==1')
            # print('>>cc = \n',len(cc))
            # pospairs_file = f'{svfolder}4932_tmpposallpairs.pkl'
            pairs.to_pickle(pospairs_file)
        else:
            pairs = pd.read_pickle(pospairs_file)
            pairs = pairs.astype(int)

        notnegpairs_file = f'{svfolder}4932_tmpnotnegpairs.pkl'
        if not os.path.exists(allpairs_file):
            notnegpairs =pd.DataFrame(columns=['p1','p2','label'])
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
                        notnegpairs = notnegpairs.append(pd.DataFrame({'p1':[p1index],'p2':[p2index],'label':[label]}))
                    else:
                        notnegpairs = notnegpairs.append(pd.DataFrame({'p1':[p2index],'p2':[p1index],'label':[label]}))
                    cn+=1

                    #240700
                    if cn % 2407 == 0:
                        tn = time.asctime()
                        print('cn time=',tn)
                        print('notnegpairs ..\t',cn/240700*100,"%",flush=True)
            # notnegpairs_file = f'{svfolder}4932_tmpnotnegpairs.pkl'
            notnegpairs.to_pickle(notnegpairs_file)
        else:
            notnegpairs = pd.read_pickle(notnegpairs_file)
            

        # val = 40
        # genenewneg = notnegpairs.query('p1== @val | p2==@val ')
        # ll = set(genenewneg['p2'])|set(genenewneg['p1'])
        # print('>>>1===\n',ll)
        # ll.remove(val)
        # print('>>>1===\n',ll)
        # print('>>>1===\n',set(genenewneg))
        # genenewneg = notnegpairs.query(' p2==@val')
        # print('>>>2===\n',genenewneg)
            
        totalrange = len(self.index2name)
        # totalindexlist= {x for x in range(0,totorange)}
        # print('totalist==\n',totalindexlist)

        def randomgene(notset:set,rangenumber,totalnumber):
            num =[]
            for i in range(0,totalnumber):
                x = random.randint(0,rangenumber)
                if x not in notset:
                    num.append(x)
            return num

        for protein in range (0,totalrange):
            # protein = 60
            # print('protein=',protein)
            if int(protein) % 1000 == 0:
                print('protein negative generate ..\t',protein)

            posdf = pairs.query('p1==@protein | p2==@protein')
            posindex = set(posdf['p2'])|set(posdf['p1'])
            # print('posindex==',posindex)
            posnumber =len(posindex)

            posindex.remove(protein)
            # print('posindex==',posindex)

            notnegdf = notnegpairs.query('p1==@protein & label==2')
            not_negindex = set(notnegdf['p2'])|set(notnegdf['p1'])
            not_negindex.remove(protein)
            # print('notngindex==',not_negindex)


            not_negindex =  posindex|not_negindex
            negindexlist = randomgene(not_negindex,totalrange,posnumber*self.negativefold)

            negdf = pd.DataFrame({'p2':negindexlist})

            negdf['p1'] = np.nan
            negdf['p1'] = negdf['p1'].fillna(protein).astype('Int64')
            # negdf['p1'] = negdf['p1'].astype('Int64')
            # print('negdf=\n',negdf)


            negdf['label'] = np.nan
            negdf['label'] = negdf['label'].fillna(0).astype('Int64')
            # print('negdf=\n',negdf)
            

            pairs = pd.merge(pairs,negdf,how='outer')
        return pairs
        # return pairs,notnegpairs


