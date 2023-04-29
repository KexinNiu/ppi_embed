


class Loaddata():
    def __init__(self,highppipath,lowppipath,esmfolder,svfolder,negativefold):
        #proteins 2 id
        #id - esm embedding
        # pos labels
        # 
        return 
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
        allinfo = allinfo.reset_index()
        print('>>>allinfo\n',allinfo[:3])
        return allinfo,index2name,name2index

    def 