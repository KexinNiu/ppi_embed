from aminoacids import to_onehot
import json
import numpy as np

class loaddataset():
    ## seq
    ## seq embedding
    ## seq ontology (class)
    def __init__(self,fastarefdictp,g2pp,pr2gp,Train=True):
        with open(fastarefdictp,'r') as f:
            self.fastadd = json.load(f)
        with open(pr2gp,'r') as f:
            self.pr2godd = json.load(f)
        with open(g2pp,'r') as f:
            self.g2ppdd = json.load(f)
            self.classes  =self.g2ppdd.keys()
        self.Train = Train
        self.load()
        self.getseqembedding()
        

    def load(self):
        
        seqidx_label=[]
        idx2prnamelist=[]
        
        for index in range (0,len(self.pr2godd.keys())-1):
            pr = list(self.pr2godd.keys())[index]
            
            prgo = self.pr2godd[pr]
            

            # prseq = self.fastadd[pr]
            idx2prnamelist.append(pr)
            
            for go in prgo:
                # print('type',type(go))
                seqidx_label.append([index,go])
                
        
        self.datainfo = seqidx_label
        self.idx2prnamelist = idx2prnamelist

        df = np.array(self.datainfo)
        # print('len df=', df.shape)
        np.random.seed(17)  
        np.random.shuffle(df)
        train, validate, test = df[0:int(.6*len(df))],df[int(.6*len(df))+1:int(.8*len(df))],df[int(.8*len(df))+1:]


        # train, validate, test = \
        #       np.split(df.sample(frac=1, random_state=42), 
        #                [int(.6*len(df)), int(.8*len(df))])
        if self.Train == True:
            self.datainfo = train
            print('len train = ',len(train))
        else:
            self.datainfo = test
            print('len test = ',len(test))

    def getseqembedding(self):
        embeddd={}
        idxembed={}
        for pr in self.fastadd.keys():
            seq = self.fastadd[pr]
            seqembed = to_onehot(seq)
            embeddd[pr] = seqembed
        for idx in range (0,len(self.idx2prnamelist)):
            seqemb = embeddd[self.idx2prnamelist[idx]]
            idxembed[str(idx)] = seqemb
        self.embed = idxembed


    def __len__(self):
        # print('len ==',len(self.datainfo))
        return len(self.datainfo)
    
    def __getitem__(self,idx):
        # print('idx=',idx)
        # seqidex,label = self.datainfo[idx]
        # seqidex = str(seqidex)
        # seqembeding = self.embed[self.idx2prnamelist[int(seqidex)]]
        return [self.embed[self.datainfo[idx][0]],self.datainfo[idx][-1]]
