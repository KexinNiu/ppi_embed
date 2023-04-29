import random
import torch
import numpy as np


# def get_data(df,pairs, newpairsflage = False):
#     data = torch.zeros((len(pairs), 2560), dtype=torch.float32)
#     labels = torch.zeros((len(pairs), 1), dtype=torch.float32)
#     # print('top df=\n',df[:5])
#     # print('lendf=',len(df),df[-5:])
#     newpairs = torch.zeros((len(pairs), 2), dtype=torch.float32)
#     for i, row in enumerate(pairs.itertuples()):
#         try:
#             f1 = df.loc[df['index']==row.p1].feature.values.tolist()[0]
#             f2 = df.loc[df['index']==row.p2].feature.values.tolist()[0]
#             f1 = torch.tensor(f1)
#             f2 = torch.tensor(f2)

#         except:
#             # print('index p1 =',row.p1)
#             # print('index p2 =',row.p2,flush=True)
#             continue

#         newf = torch.cat((f1,f2),0)
#         i1 = torch.tensor(row.p1,dtype=torch.float32).unsqueeze(axis=0)
#         i2 = torch.tensor(row.p2,dtype=torch.float32).unsqueeze(axis=0)
#         newp = torch.cat((i1,i2),0) 
#         # ff1,ff2 = newf.chunk(2)
#         data[i,:] = torch.FloatTensor(newf)
#         labels[i,:] = torch.tensor(row.label)
#         newpairs[i,:] = torch.tensor(newp)
#     if newpairsflage ==True:
#         return data,labels,newpairs
#     else:
#         return data,labels

def get_traindata(proteins_index, esm_data, pairs, notneg_interactions,ktime = 5):
    # pos:neg = 1:ktimes
    for pr1 in proteins_index:
        f1 = esm_data[pr1]
        p1pairs = pairs.query("protein1==@pr1 | protein2==@pr1")
        pr2s = np.unique(p1pairs[["protein1", "protein2"]].values)## postives
        if len(pr2s)==0:
            continue
        notp1pairs = notneg_interactions.query("protein1==@pr1 | protein2==@pr1")
        not_negp2s = np.unique(notp1pairs[["protein1", "protein2"]].values)##non negatives
        
        ## negatives candidates
        negp2s = set(proteins_index) - set(pr2s) - set(not_negp2s) - set([pr1])
        negpr2s = random.sample(list(negp2s), ktime*len(pr2s)) 

        # Get features for protein2
        label  = torch.ones((len(pr2s), 1), dtype=torch.float32)
        label_neg = torch.zeros((len(negpr2s), 1), dtype=torch.float32)


        f2data = torch.stack([esm_data[i] for i in pr2s])
        f2data_neg = torch.stack([esm_data[i] for i in negpr2s])
        
        # cat pos neg for protein2
        labels = torch.cat((label,label_neg),0)
        f2_data = torch.cat((f2data,f2data_neg),0)
        f1_data = torch.stack([f1 for i in range(len(labels))])
        try:
            final_f1 = torch.cat((final_f1,f1_data),0)
            final_f2 = torch.cat((final_f2,f2_data),0)
            final_label = torch.cat((final_label,labels),0)
        except:
            final_f1 = f1_data
            final_f2 = f2_data
            final_label = labels
    
    # print('>>>Loading traindata ....\n', flush=True)
    print('final_f1==={}\n'.format(final_f1.shape))
    print('final_f2==={}\n'.format(final_f2.shape))
    # print('final_label==={}\n{}'.format(final_label.shape,final_label[:10]))
    return final_f1,final_f2,final_label

def get_val_testdata(proteins_index, esm_data, pairs, notneg_interactions):
    # pos label ==1
    # not neg label ==2
    # remaining # random generated ==3
    count=0
    for pr1 in proteins_index:
        f1 = esm_data[pr1]
        p1pairs = pairs.query("protein1==@pr1 | protein2==@pr1")
        pr2s = np.unique(p1pairs[["protein1", "protein2"]].values)## postives
        notp1pairs = notneg_interactions.query("protein1==@pr1 | protein2==@pr1")
        not_negp2s = np.unique(notp1pairs[["protein1", "protein2"]].values)##not negatives
        
        ## negatives candidates
        remaining_unlabeled = set(proteins_index) - set(pr2s) - set(not_negp2s) - set([pr1]) # all negatives candidates
        not_negp2s = list(not_negp2s)
        remaining_unlabeled = list(remaining_unlabeled)
        pr2s = list(pr2s)

        # Get features for protein2
        label  = torch.ones((len(pr2s), 1), dtype=torch.float32)
        label_r = torch.full((len(remaining_unlabeled), 1),fill_value=3.0, dtype=torch.float32) ## remaining 
        label_notneg = torch.full((len(not_negp2s), 1),fill_value=2.0,dtype=torch.float32)
        
        f2data_r = torch.stack([esm_data[i] for i in remaining_unlabeled])# all negatives candidates
       
        try:
            f2data = torch.stack([esm_data[i] for i in pr2s])## postives  
        except:
            f2data = torch.empty_like(f2data_r)
        try:
            f2data_notneg = torch.stack([esm_data[i] for i in not_negp2s])##not negatives
        except:
            f2data_notneg = torch.empty_like(f2data_r)

        labels = torch.cat((label,label_r,label_notneg),0)
        f2_data = torch.cat((f2data,f2data_r,f2data_notneg),0)
        f1_data = torch.stack([f1 for i in range(len(labels))]) # pos+notneg+r

        # if len(pr2s)!=0:
        #     f2data = torch.stack([esm_data[i] for i in pr2s])## postives           
        #     if len(not_negp2s)!=0:
        #         f2data_notneg = torch.stack([esm_data[i] for i in not_negp2s])##not negatives
        #         labels = torch.cat((label,label_r,label_notneg),0)
        #         f2_data = torch.cat((f2data,f2data_r,f2data_notneg),0)
        #         f1_data = torch.stack([f1 for i in range(len(labels))]) # pos+notneg+r
               
        #     else:
        #         labels = torch.cat((label,label_r),0)
        #         f2_data = torch.cat((f2data,f2data_r),0)
        #         f1_data = torch.stack([f1 for i in range(len(labels))])# pos+r
                
        # else:
        #     if len(not_negp2s)!=0:
        #         f2data_notneg = torch.stack([esm_data[i] for i in not_negp2s])##not negatives
        #         labels = torch.cat((label_r,label_notneg),0)
        #         f2_data = torch.cat((f2data_r,f2data_notneg),0)
        #         f1_data = torch.stack([f1 for i in range(len(labels))]) #notneg+r  
        #     else:
        #         labels = label_r
        #         f2_data = f2data_r
        #         f1_data = torch.stack([f1 for i in range(len(labels))]) #r
                
        
        f2_index = torch.tensor([i for i in list(pr2s)+list(remaining_unlabeled)+list(not_negp2s)],dtype = torch.float32)
        f1_index = torch.tensor([pr1 for i in range(len(labels))],dtype = torch.float32)

        try:
            final_f1 = torch.cat((final_f1,f1_data),0)
            final_f2 = torch.cat((final_f2,f2_data),0)
            final_label = torch.cat((final_label,labels),0)
        except:
            final_f1 = f1_data
            final_f2 = f2_data
            final_label = labels
        count+= 1
        if count % 200==0:
            print('count={}'.format(count),flush=True)

    print('len(labels)={}'.format(final_label.shape),flush=True)
    print('len(final_f1)={}'.format(final_f1.shape),flush=True)
    print('len(final_f2)={}'.format(final_f2.shape),flush=True)
    print('len(f1_index)={}'.format(f1_index.shape),flush=True)
    print('len(f2_index)={}'.format(f2_index.shape),flush=True)

    return final_f1,final_f2,final_label,f1_index,f2_index

def get_index_val_testdata(proteins_index, esm_data, pairs, notneg_interactions):
    count=0
    for pr1 in proteins_index:
        f1 = esm_data[pr1]
        p1pairs = pairs.query("protein1==@pr1 | protein2==@pr1")
        pr2s = np.unique(p1pairs[["protein1", "protein2"]].values)## postives
        notp1pairs = notneg_interactions.query("protein1==@pr1 | protein2==@pr1")
        not_negp2s = np.unique(notp1pairs[["protein1", "protein2"]].values)##not negatives
        
        ## negatives candidates
        remaining_unlabeled = set(proteins_index) - set(pr2s) - set(not_negp2s) - set([pr1]) # all negatives candidates
        not_negp2s = list(not_negp2s)
        remaining_unlabeled = list(remaining_unlabeled)
        pr2s = list(pr2s)

        # Get features for protein2
        label  = torch.ones((len(pr2s), 1), dtype=torch.float32)
        label_r = torch.full((len(remaining_unlabeled), 1),fill_value=3.0, dtype=torch.float32) ## remaining 
        label_notneg = torch.full((len(not_negp2s), 1),fill_value=2.0,dtype=torch.float32)
        
        # f2data_r = torch.stack([esm_data[i] for i in remaining_unlabeled])# all negatives candidates
       
        # try:
        #     f2data = torch.stack([esm_data[i] for i in pr2s])## postives  
        # except:
        #     f2data = torch.empty_like(f2data_r)
        # try:
        #     f2data_notneg = torch.stack([esm_data[i] for i in not_negp2s])##not negatives
        # except:
        #     f2data_notneg = torch.empty_like(f2data_r)

        labels = torch.cat((label,label_r,label_notneg),0)
        # f2_data = torch.cat((f2data,f2data_r,f2data_notneg),0)
        # f1_data = torch.stack([f1 for i in range(len(labels))]) # pos+notneg+r
    
        f2_index = torch.tensor([i for i in list(pr2s)+list(remaining_unlabeled)+list(not_negp2s)],dtype = torch.float32)
        f1_index = torch.tensor([pr1 for i in range(len(labels))],dtype = torch.float32)

        try:
            # final_f1 = torch.cat((final_f1,f1_data),0)
            # final_f2 = torch.cat((final_f2,f2_data),0)
            final_label = torch.cat((final_label,labels),0)
            final_f1index = torch.cat((final_f1index,f1_index),0)
            final_f2index = torch.cat((final_f2index,f2_index),0)
        except:
            # final_f1 = f1_data
            # final_f2 = f2_data
            final_label = labels
            final_f1index = f1_index
            final_f2index = f2_index
        count+= 1
        if count % 200==0:
            print('count={}'.format(count),flush=True)

    print('len(labels)={}'.format(final_label.shape),flush=True)     
    print('len(final_f1index)={}'.format(final_f2index.shape),flush=True)
    print('for i in batchindex,',final_f1index[5].item())
    print('len(final_f2index)={}'.format(final_f2index.shape),flush=True)
  
    return final_label,final_f1index,final_f2index

def get_index_traindata(proteins_index, esm_data, pairs, notneg_interactions,ktime = 5):
    # pos:neg = 1:ktimes
    print('total train protein={}'.format(len(proteins_index)),flush=True)
    print('K times ={}'.format(ktime),flush=True)
    for pr1 in proteins_index:
        f1 = esm_data[pr1]
        p1pairs = pairs.query("protein1==@pr1 | protein2==@pr1")
        # print('total train pairs={}'.format(len(p1pairs)),flush=True)

        pr2s = np.unique(p1pairs[["protein1", "protein2"]].values)## postives
        if len(pr2s)==0:
            continue
        notp1pairs = notneg_interactions.query("protein1==@pr1 | protein2==@pr1")
        not_negp2s = np.unique(notp1pairs[["protein1", "protein2"]].values)##non negatives
        
        ## negatives candidates
        negp2s = set(proteins_index) - set(pr2s) - set(not_negp2s) - set([pr1])
        negcount = min(ktime*len(pr2s),len(list(negp2s)))
        negpr2s = random.sample(list(negp2s), negcount) 

        # Get features for protein2
        label  = torch.ones((len(pr2s), 1), dtype=torch.float32)
        label_neg = torch.zeros((len(negpr2s), 1), dtype=torch.float32)
        


        # f2data = torch.stack([esm_data[i] for i in pr2s])
        # f2data_neg = torch.stack([esm_data[i] for i in negpr2s])
        
        # cat pos neg for protein2
        labels = torch.cat((label,label_neg),0)
        f2_index = torch.tensor([i for i in list(pr2s)+list(negpr2s)],dtype = torch.float32)
        f1_index = torch.tensor([pr1 for i in range(len(labels))],dtype = torch.float32)
        # f2_data = torch.cat((f2data,f2data_neg),0)
        # f1_data = torch.stack([f1 for i in range(len(labels))])
        try:
            # final_f1 = torch.cat((final_f1,f1_data),0)
            # final_f2 = torch.cat((final_f2,f2_data),0)
            final_label = torch.cat((final_label,labels),0)
            final_f1index = torch.cat((final_f1index,f1_index),0)
            final_f2index = torch.cat((final_f2index,f2_index),0)
        except:
            # final_f1 = f1_data
            # final_f2 = f2_data
            final_label = labels
            final_f1index = f1_index
            final_f2index = f2_index
    
    print('>>>Loading traindata ....\n', flush=True)
    print('final_f1==={}\n'.format(final_f1index.shape))
    print('final_f2==={}\n'.format(final_f2index.shape))
    # print('final_label==={}\n{}'.format(final_label.shape,final_label[:10]))
    return final_label,final_f1index,final_f2index
