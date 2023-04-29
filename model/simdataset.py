import json
import glob
import random
import torch
import pandas as pd

# ppin2esmnp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.protein.aliases.v11.5.1.json'
# ppip='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/4932.finalpairs.txt'
class dataset_pairwise():
    def __init__(self,ppipath,esmfolder,svfolder,Train:str):
        self.esmfolder = esmfolder
        self.svfolder= svfolder
        self.pos_negp=svfolder+'perppos_neg.json'
        self.Train = Train
        if self.Train =='train':
            self.ppipath =ppipath
            self.loadppi2pos_neg()
            self.loadesmfolder() 
        else:
            self.allpairs=ppipath
            self.splitallpairs()

    
    def loadppi2pos_neg(self):
        # with open(self.alipath,'r') as f:
        #     alidd = json.load(f)
        ddpos_neg={'0':{},'1':{}}
        pairs=[]
        
        with open(self.ppipath,'r') as f:
            for line in f:
                line = line.strip('\n')
                p1,p2,label = line.split('\t')
                pairs.append((p1,p2))
                if label =='1':
                    try:
                        ddpos_neg['1'][p1].append(p2)
                    except:
                        ddpos_neg['1'][p1]=[p2]
                    try:
                        ddpos_neg['1'][p2].append(p1)
                    except:
                        ddpos_neg['1'][p2]=[p1]
                else:
                    try:
                        ddpos_neg['0'][p1].append(p2)
                    except:
                        ddpos_neg['0'][p1]=[p2]
                    try:
                        ddpos_neg['0'][p2].append(p1)
                    except:
                        ddpos_neg['0'][p2]=[p1]
        self.allpairs = pairs
        self.ddpos_neg = ddpos_neg
        # with open(self.pos_negp,'w') as f:
        #     json.dump(self.ddpos_neg,f)

        # self.pairs = pairs
        train, validate, test = pairs[0:int(.6*len(pairs))],pairs[int(.6*len(pairs))+1:int(.8*len(pairs))],pairs[int(.8*len(pairs))+1:]
        
        self.pairs = train
        
        return

    def loadesmfolder(self):
        orip = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
        # esmfolderpath =''
        feature=[]
        rnames = []
        n2f = {}

        pp = orip+'*'
        for filename in glob.glob(pp):
            name = filename.replace(orip,'')
            name = name.replace('.pt','')
            rnames.append(name)
            # filename = filename.replace(orip,esmfolderpath)
            fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
            feature.append(fff)
            n2f[name] = fff
        self.prname=rnames
        self.features=feature
        n2f = pd.DataFrame.from_dict(n2f, orient='tight')
        self.n2f = n2f
        print(n2f[:10])
        return

    def splitallpairs(self):
        # self.allpairs
        _, validate, test = self.allpairs[0:int(.6*len(self.allpairs))],self.allpairs[int(.6*len(self.allpairs))+1:int(.8*len(self.allpairs))],self.allpairs[int(.8*len(self.allpairs))+1:]
        if self.Train=='val':
            self.pairs = validate
        elif self.Train=='test':
            self.pairs = test

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,idx):
        ##feature --> self.n2f 
        ##pos_neg --> self.ddpos_neg
        return [self.pairs[idx]]


class dataset_perprotein():
    def __init__(self,ppipath,esmfolder,svfolder):
        self.esmfolder = esmfolder
        self.svfolder= svfolder
        self.pos_negp=svfolder+'perppos_neg.json'
        self.ppipath =ppipath
        self.loadppi2pos_neg()
        self.loadesmfolder() 
        # else:
        #     self.allpairs=ppipath
        #     self.splitallpairs()
    # def load

    def loadesmfolder(self):
        orip = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/esm2_sgd/'
        feature=[]
        rnames = []
        n2f = {}
        pp = orip+'*'
        for filename in glob.glob(pp):
            name = filename.replace(orip,'')
            name = name.replace('.pt','')
            rnames.append(name)
            # filename = filename.replace(orip,esmfolderpath)
            fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
            feature.append(fff)
            n2f[name] = fff
        self.prname=rnames
        self.features=feature
        self.n2f = n2f
        print('dim =',len(self.features[0]))
        return


    def loadppi2pos_neg(self):
        ddpos_neg={'0':{},'1':{}}
        pairs=[]
        
        with open(self.ppipath,'r') as f:
            for line in f:
                line = line.strip('\n')
                p1,p2,label = line.split('\t')
                # if p1<p2:
                #     pass
                # else:
                #     tmp = p1
                #     p1  =p2
                #     p2 = tmp
                pairs.append((p1,p2))
                if label =='1':
                    try:
                        ddpos_neg['1'][p1].append(p2)
                    except:
                        ddpos_neg['1'][p1]=[p2]
                    try:
                        ddpos_neg['1'][p2].append(p1)
                    except:
                        ddpos_neg['1'][p2]=[p1]
                else:
                    try:
                        ddpos_neg['0'][p1].append(p2)
                    except:
                        ddpos_neg['0'][p1]=[p2]
                    try:
                        ddpos_neg['0'][p2].append(p1)
                    except:
                        ddpos_neg['0'][p2]=[p1]
        self.allpairs = pairs
        self.ddpos_neg = ddpos_neg


    def __len__(self):
        return len(self.prname)

    def __getitem__(self,idx):
        ##feature --> self.n2f 
        ##pos_neg --> self.ddpos_neg
        return [self.prname[idx]]

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, random_state=0)
y_hat_all = []
for train_index, test_index in kf.split(X, y):
    reg = RandomForestRegressor(n_estimators=50, random_state=0)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = reg.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    y_hat_all.append(y_hat)

# train, test = train_test_split(df, test_size=0.2)
class dataset_prosplit():
    def __init__(self,allnames,allpairs,highppipath,lowppipath,esmfolder,svfolder,datalabel:str):
        self.esmfolder = esmfolder
        self.svfolder= svfolder
        self.hppipath =highppipath 
        ## if not train ,highppipath is protein names
        self.lppipath =lowppipath  
        self.label = datalabel

        if datalabel =='train':
            allinfo = self.loadesmfolder()

            feature = n2f['feature']
            rnames = n2f['name']
            print('features = ',type(feature),feature[:10])
            print('rnames = ',type(rnames),rnames[:10])
            self.allnames=rnames
            self.features=feature
            print('dim =',len(self.features[0]))

            self.allinfo = allinfo

            random.seed(17)
            random.shuffle(rnames)
            train_names = rnames[:round(len(rnames)*0.6)]
            self.load_allpairs()
            self.proteins = train_names
            self.posdd = self.getposdd()
        elif datalabel =='val':
            # self.allnames = allnames
            self.allpairs = allpairs
            random.seed(17)
            random.shuffle(allnames)
            val_names = allnames[round(len(allnames)*0.6)+1:round(len(allnames)*0.8)]
            self.proteins = val_names
            self.posdd = self.getposdd()
        elif datalabel =='test':
            # self.allnames = allnames
            self.allpairs = allpairs
            random.seed(17)
            random.shuffle(allnames)
            test_names = rnames[round(len(allnames)*0.8)+1:]
            self.proteins = test_names
            self.posdd = self.getposdd()

    def loadesmfolder(self):
        orip = self.esmfolder
        feature=[]
        rnames = []
        n2f = {}
        pp = orip+'*'
        for filename in glob.glob(pp):
            name = filename.replace(orip,'')
            name = name.replace('.pt','')
            rnames.append(name)
            # filename = filename.replace(orip,esmfolderpath)
            fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
            feature.append(fff)
            n2f[name] = fff
            if len(feature) >50:
                break
        # self.prname=rnames
        # self.features=feature
        # self.n2f = n2f
        n2f = pd.DataFrame(n2f.items(), columns=['name', 'feature'])
        # print('n2f::',type(n2f),n2f[:10])
        # print("n2f['name\n", n2f['name'])
        # print("n2f['feature'][:5]\n", n2f['feature'][:5])
        # a ='4932.YKL124W'/
        # print('取一行\n',n2f.loc[n2f['name']==a],flush=True)
        '''0    4932.YKL124W  [0.03605465218424797, -0.02259356901049614, 0..'''
        feature = n2f['feature']
        rnames = n2f['name']
        print('features = ',type(feature),feature[:10])
        print('rnames = ',type(rnames),rnames[:10])

        return n2f
    
    
    def load_allpairs(self):
        pairs=[]
        with open(self.hppipath,'r') as f:
            for line in f:
                line = line.strip('\n')
                p1,p2,label = line.split('\t')
                
                if p1<p2:
                    pass
                else:
                    tmp = p1
                    p1  =p2
                    p2 = tmp
                pairs.append((p1,p2))
        self.allpairs = pairs

        notnegpairs=[]
        with open(self.lppipath,'r') as f:
            for line in f:
                line = line.strip('\n')
                p1,p2,label = line.split('\t')
                
                if p1<p2:
                    pass
                else:
                    tmp = p1
                    p1  =p2
                    p2 = tmp
                notnegpairs.append((p1,p2))
        self.notneg = notnegpairs

    def getposdd(self):
        ddpos = {}
        proteins = set(self.proteins)

        print('label=',self.label)
        for p1,p2 in self.allpairs:
            if p1 in proteins and p2 in proteins:
                try:
                    ddpos[p1].append(p2)
                except:
                    ddpos[p1]=[p2]
                try:
                    ddpos[p2].append(p1)
                except:
                    ddpos[p2]=[p1]
        # self.posdd = ddpos
        # print('1type trainposdd=',type(ddpos))
        return ddpos

        
    def __len__(self):
        return len(self.proteins)
    def __getitem__(self,idx):
        # print('self.proteins[idx]=',self.proteins[idx])
        return self.proteins[idx]