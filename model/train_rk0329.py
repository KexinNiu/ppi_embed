



import os
@ck.command()

@ck.option(
    '--data-root', '-dr', default='data',
    help='ppi and interactions folder root')
@ck.option(
    '--batch-size', '-bs', default=16,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=32,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
@ck.option(
    '--hiddim', '-hd', default=100,
    help='hidden dimension of the model') 
@ck.option(
    '--negativefold', '-nf', default=4,
    help='negativefold , negatives is x fold of positives') 
@ck.option(
    '--name', '-n', default='currname',
    help='names of file') 
@ck.option(
    '--esmfolder', '-esmf',
    help='esmfolder of esm2') 
@ck.option(
    '--modelname', '-mn',
    help='name of model,onef,twof')
@ck.option(
    '--metric', '-metric',
    default = 'dot',
    help='options =')
    # ['dot','Euclidean distance','Manhattan metric','Chebychev metric']  
@ck.option(
    '--learningrate', '-lr',default = 1e-4,
    help='learning rate') 


def main(data_root, batch_size, epochs, hiddim, load, device, negativefold,name,esmfolder, modelname,learningrate,metric):
    print('>>>',torch.cuda.is_available())
    lr= learningrate
    print('lr=',lr,flush=True)
    print('batch_size=',batch_size,flush=True)

    model_file = f'{data_root}/model_esm2_withabs{epochs}_{modelname}_{metric}_{batch_size}.th'
    out_file = f'{data_root}/predictions_{epochs}_{modelname}_{metric}_{batch_size}.pkl'

    allinfo_file = f'{data_root}/{name}_allinfo.pkl'
    allpairs_file = f'{data_root}/{name}_allpairs.pkl'
    
    allpairs_csvfile = f'{data_root}/{name}_allpairs.csv'

    name2index_file = f'{data_root}/{name}_name2index.pkl'
    index2name_file = f'{data_root}/{name}_index2name.pkl'
    high_ppipath = f'{data_root}/{name}.highconf.links.txt'
    low_ppipath = f'{data_root}/{name}.lowconf.links.txt'
    esmfolder = esmfolder
    svfolder = f'{data_root}/'
    

    if not os.path.exists(allpairs_csvfile):
        print('>>>!!not exists',flush=True)
        dataset = Loaddata(high_ppipath,low_ppipath,esmfolder,svfolder,negativefold,name)
        allinfo = dataset.allinfo 
        allpairs = dataset.allpairs
        name2index = dataset.name2index
        index2name = dataset.index2name
        print('>>>sv...',flush=True)
        print('>>>finished sv',flush=True)

    else:

        print('\n\n >>>load...',flush=True)
        allinfo = pd.read_pickle(allinfo_file)
        # allpairs = pd.read_pickle(allpairs_file)
        allpairs = pd.read_csv(allpairs_csvfile)
        # print('>>>allpairs top5\n',allpairs[:2],flush=True)
        allpairs = allpairs.astype(int)
        with open(name2index_file,'r')as f:
            name2index = json.load(f)

        print('>>>finished loading\n',flush=True)

    if modelname == 'onef':
        net = Featuredotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280) 
    elif modelname =='twof':
        net = twoFeaturedotmodel(in_dim=1280,hidden_dim=hiddim,out_dim=1280) 

    loss_func = nn.LogSigmoid()
    # loss_func = nn.BCELoss()
    # lr = lr
    print('lr=',lr,flush=True)
    # optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[5,], gamma=0.1)
    print('>>>model setting done\n',flush=True)

    
    # model 
    ########################################

    kf = KFold(n_splits = 5, shuffle = True, random_state = 17)
    for i, (train_index, test_index) in enumerate(kf.split(allinfo)):
        print(f"Fold {i}:",flush=True)
        # if i ==1:
        #     break
        valid_index = train_index[:round(len(train_index)*0.25)]
        train_index1 = list(set(train_index).difference(set(train_index[:round(len(train_index)*0.25)])))
       
        train_data= allinfo.iloc[train_index1]
        test_data= allinfo.iloc[test_index]
        valid_data = allinfo.iloc[valid_index]

        train_pairs = allpairs[(allpairs['p1'].isin(train_index1)) & (allpairs['p2'].isin(train_index1))]
        test_pairs = allpairs[(allpairs['p1'].isin(test_index)) & (allpairs['p2'].isin(test_index))]
        valid_pairs = allpairs[(allpairs['p1'].isin(valid_index)) & (allpairs['p2'].isin(valid_index))]
        def make_rankingpairs(train_pairs)

        traininfo_file = f'{data_root}/{name}_{i}_traininfo.pkl'
        trainpairs_file = f'{data_root}/{name}_{i}_trainpairs.pkl'

        train_data.to_pickle(traininfo_file)
        train_pairs.to_pickle(trainpairs_file)

        traindata = get_data(train_data,train_pairs)
        train_loader = FastTensorDataLoader(
            *traindata, batch_size=batch_size, shuffle=True)
        
        validdata = get_data(valid_data,valid_pairs)

        valid_loader = FastTensorDataLoader(
            *validdata, batch_size=batch_size, shuffle=True)
        # data prepare##
        print(' # data prepare##',flush=True)
        #####################################################
        

        
        best_loss = 100000.0
        if not load:
            net.train()
            print('Training the model from the begnning',flush=True)
            for epoch in range(epochs):
                train_loss = 0
                pt =0
                train_steps = int(math.ceil(len(train_data) / batch_size))
                with ck.progressbar(length=train_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in train_loader:
                        bar.update(1)
                        pt+=1
                        batch_labels =  torch.squeeze(batch_labels)
                        output = net(batch_features)
                        
                        try:
                            loss = F.binary_cross_entropy(output, batch_labels)
                        except:
                            batch_labels = torch.unsqueeze(batch_labels, 0)
                            loss = F.binary_cross_entropy(output, batch_labels)
                            
                        total_loss = loss
                        train_loss += loss.detach().item()
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        if pt % 500 ==0:
                            print('batch loss=',total_loss.item(),flush=True)
                        if pt %2500==0 and epoch >=25:
                            print('ouput={}\nbatch_labels={}'.format(output[:5],batch_labels[:5]))
                train_loss /= train_steps
                print('>>Validation',flush=True)
                net.eval()
                with torch.no_grad():
                    valid_steps = int(math.ceil(len(valid_data) / batch_size))
                    valid_loss = 0
                    preds = []
                    alllabel = []

                    with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                        for batch_features, batch_labels in valid_loader:
                            bar.update(1)
                            # batch_features = batch_features.to(device)
                            # batch_labels = batch_labels.to(device)
                            # batch_features = batch_features
                            # batch_labels = batch_labels
                            batch_labels =  torch.squeeze(batch_labels)
                            logits = net(batch_features)
                            batch_loss = F.binary_cross_entropy(logits, batch_labels)
                            valid_loss += batch_loss.detach().item()
                            preds = numpy.append(preds, logits.detach().cpu().numpy())
                            alllabel = numpy.append(alllabel, batch_labels.detach().cpu().numpy())

                    valid_loss /= valid_steps
                    # print('labels = validata[1]={}'.format(validdata[1]),flush=True)
                    # print('pred={}'.format(preds))
                    # roc_auc = compute_roc(validdata[1], preds)
                    # print('roc_auc=',roc_auc)
                    # print('alllabel=',alllabel)

                    roc_auc = compute_roc(alllabel, preds)
                    print('roc_auc1=',roc_auc)

                    print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}',flush=True)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    print('Saving model',flush=True)
                    torch.save(net.state_dict(), model_file)

                scheduler.step()

        # Loading best model
        print('..Loading the best model')
        net.load_state_dict(torch.load(model_file))
        net.eval()
        ####
        testdata = get_data(test_data,test_pairs,True)
        test_loader = FastTensorDataLoader(
            *testdata, batch_size=batch_size, shuffle=True)
        
        ####

        with torch.no_grad():
            test_steps = int(math.ceil(len(testdata[1]) / batch_size))
            test_loss = 0
            preds = []
            alllabel = []
            alltestpairs =[]
            
            with ck.progressbar(length=test_steps, show_pos=True) as bar:
                for batch_features, batch_labels,bathch_indexs in test_loader:
                    bar.update(1)
                    # batch_features = batch_features.to(device)
                    # batch_labels = batch_labels.to(device)
                    batch_labels =  torch.squeeze(batch_labels)
                    # print('batch_labels=',batch_labels,flush=True)
                    # alllabel.append(batch_labels.item())
                    logits = net(batch_features)
                    batch_loss = F.binary_cross_entropy(logits, batch_labels)
                    test_loss += batch_loss.detach().cpu().item()
                    preds = numpy.append(preds, logits.detach().cpu().numpy())
                    alllabel = numpy.append(alllabel, batch_labels.detach().cpu().numpy())
                    
                    ###pairs:
                    x0index,x1index = bathch_indexs.chunk(2,axis=1)
                    # print('x0index=',x0index[:3])
                    x0index = torch.squeeze(x0index).tolist()
                    x1index = torch.squeeze(x1index).tolist()
                    pp = zip(x0index,x1index)
                    for p in pp:
                        alltestpairs.append(p)
                    # alltestpairs = numpy.append(alltestpairs, bathch_indexs.detach().cpu().numpy())
                    ######
                
                
                test_loss /= test_steps
            # preds = preds.reshape(-1, n_terms)?????? 
            # print('alllabel ={}'.format(alltestpairs),flush=True)
            # roc_auc = compute_roc(testdata[1], preds)
            roc_auc = compute_roc(alllabel, preds)
            # print('all label roc_auc={}'.format(roc_auc1))

            print(f'Test Loss - {test_loss}, AUC - {roc_auc}',flush=True)
        # test_pairs['preds'] = preds
        predresult = pd.DataFrame({
            "pairs": alltestpairs,
            "labels": alllabel,
            "preds":preds
        })
        predresult.to_pickle(out_file)
        # test_pairs.to_pickle(out_file)

def get_data(df,pairs, newpairsflage = False):
    data = torch.zeros((len(pairs), 2560), dtype=torch.float32)
    labels = torch.zeros((len(pairs), 1), dtype=torch.float32)
    # print('top df=\n',df[:5])
    # print('lendf=',len(df),df[-5:])
    newpairs = torch.zeros((len(pairs), 2), dtype=torch.float32)
    for i, row in enumerate(pairs.itertuples()):
        try:
            f1 = df.loc[df['index']==row.p1].feature.values.tolist()[0]
            f2 = df.loc[df['index']==row.p2].feature.values.tolist()[0]
            f1 = torch.tensor(f1)
            f2 = torch.tensor(f2)

        except:
            # print('index p1 =',row.p1)
            # print('index p2 =',row.p2,flush=True)
            continue

        newf = torch.cat((f1,f2),0)
        i1 = torch.tensor(row.p1,dtype=torch.float32).unsqueeze(axis=0)
        i2 = torch.tensor(row.p2,dtype=torch.float32).unsqueeze(axis=0)
        newp = torch.cat((i1,i2),0) 
        # ff1,ff2 = newf.chunk(2)
        data[i,:] = torch.FloatTensor(newf)
        labels[i,:] = torch.tensor(row.label)
        newpairs[i,:] = torch.tensor(newp)
    if newpairsflage ==True:
        return data,labels,newpairs
    else:
        return data,labels

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    # print('labels =={}'.format(labels),flush=True)
    # print('pred={}'.format(preds),flush=True)
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    # print('fpr={}\ntpr={}'.format(fpr,tpr))
    roc_auc = auc(fpr, tpr)
    # print('rocauc={}'.format(roc_auc))


    return roc_auc



class Featuredotmodel(nn.Module):
    # def __init__(self,in_dim:int,hidden_dim:int, out_dim:int,device:str):
    def __init__(self,in_dim:int,hidden_dim:int, out_dim:int):
        super().__init__()
        self.__in_dim = in_dim
        self.__out_dim = out_dim
        self.__hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.Sigmoid = nn.Sigmoid()
        # self.feature_norm = nn.LayerNorm()

    def forward(self,x:torch.Tensor)-> torch.Tensor:

        x0,x1 = x.chunk(2,axis=1)
        x0 = self.linear1(x0)
        x0 = self.Sigmoid(x0)
        x0 = self.linear2(x0)
        dotproduct = torch.bmm(
            x0.view(x0.shape[0],1,x0.shape[1]),
            x1.view(x0.shape[0],x0.shape[1],1),
        ).reshape(-1)
        
        # print('dotproduct',dotproduct.size(),flush=True)
        # dotproduct = torch.abs(dotproduct)
        # output = self.Sigmoid(dotproduct)
        output = dotproduct
        return output

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
        dotproduct = torch.abs(dotproduct)

        output = self.Sigmoid(dotproduct)
        return output
