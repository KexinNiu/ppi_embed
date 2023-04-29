
import pandas
import torch
import glob




def get_posdf(alldf,allpairs,notnegpairs):
    interactions = pandas.DataFrame(columns=alldf['name'],index=alldf['name'])
    for p1,p2 in allpairs:
        interactions.loc[p1,p2] = 1
        interactions.loc[p2,p1] = 1
    for pp1,pp2 in  notnegpairs:
        interactions.loc[pp1,pp2] = -1
        interactions.loc[pp2,pp1] = -1
    return interactions



def gene_pos(prlist,allpairs):
    # input a list of protein lists
    # output a list of dict  
    prsetlist= prlist

    ddposlist=[]
    for i in range(0,len(prlist)):
        # prset = set(prlist[i])
        # prsetlist.append(prset)
        ddposlist.append({})
    for p1,p2 in allpairs:
        for i in range(0,len(prlist)):
            proteins = prsetlist[i]
            # ddpos = ddposlist[i]
            if p1 in proteins and p2 in proteins:
                try:
                    ddposlist[i][p1].append(p2)
                except:
                    ddposlist[i][p1]=[p2]
                try:
                    ddposlist[i][p2].append(p1)
                except:
                    ddposlist[i][p2]=[p1]
    return ddposlist
    
        
    return #dataframe name, interact proteins

# def gene_random_neg(batchpr,notneg,allinfo):
#     f0 = torch.tensor([])
#     for pr in batchpr:

#     return # dataframe 
# def gene_neg(prlist, allnotneg,fold):
#     negddlist=[]
#     return negddlist

class Loaddata():
    def __init__(self,highppipath,lowppipath,esmfolder,svfolder) :
        self.esmfolder = esmfolder
        self.svfolder = svfolder
        self.hppipath = highppipath 
        self.lppipath = lowppipath  
        # self.label = datalabel

        allinfo = self.loadesmfolder()
        self.allinfo = allinfo
        # feature = n2f['feature']/
        # rnames = n2f['name']
        allpairs,notnegpairs = self.loadallpairs()
        self.allpairs = allpairs
        self.notnegpairs = notnegpairs

    def loadesmfolder(self):
        dd ={}
        for filename in glob.glob(self.esmfolder+'*'):
            name = filename.replace(self.esmfolder,'')
            name = name.replace('.pt','')
            fff = torch.load(filename)['mean_representations'][33].detach().numpy().tolist()
            dd[name]=fff
            if len(list(dd.keys())) >50:
                break
        allinfo = pandas.DataFrame(dd.items(),columns=['name', 'feature'])
        return allinfo

    def loadallpairs(self):
        pairs = []
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
        return pairs,notnegpairs







