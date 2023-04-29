import torch
import torch.nn as nn
import torch.optim as optim

# class ProteinInteractionModel(nn.Module):
#     def __init__(self, input_size):
#         super(ProteinInteractionModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = nn.functional.relu(x)
#         x = self.fc2(x)
#         x = nn.functional.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x

# # Define the positive-unlabeled learning loss function
# def PULoss(outputs, alpha=0.5, beta=0.5):
#     # Calculate the loss for positive examples
#     loss_positive = -alpha * torch.mean(torch.log(outputs))

#     # Calculate the loss for negative examples
#     loss_negative = -beta * torch.mean(torch.log(1 - outputs))

#     # Total loss
#     loss = loss_positive + loss_negative

#     return loss

# # Load your labeled protein interaction data
# # X_pos is a tensor of size (n_pos_samples, 2, input_size)
# # where X_pos[i,0] and X_pos[i,1] represent the vectors for two proteins in the i-th positive interaction pair
# X_pos = ... # Load your positive input data here

# # Create all possible combinations of protein pairs
# n_pos_samples = X_pos.shape[0]
# X_neg = torch.zeros((n_pos_samples * n_pos_samples, 2, input_size))
# for i in range(n_pos_samples):
#     for j in range(n_pos_samples):
#         if i != j:
#             X_neg[i * n_pos_samples + j] = torch.stack([X_pos[i, 0], X_pos[j, 1]])

# # Create PyTorch DataLoader for your training set
# train_dataset = torch.utils.data.ConcatDataset([
#     torch.utils.data.TensorDataset(X_pos, torch.ones(n_pos_samples)),
#     torch.utils.data.TensorDataset(X_neg, torch.zeros(n_pos_samples * n_pos_samples))
# ])
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# # Create an instance of your protein interaction model
# input_size = 1280
# model = ProteinInteractionModel(input_size)

# # Define the optimizer and learning rate scheduler
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# # Define the evaluation function
# def evaluate(model, dataloader):
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#     model.eval()
#     with torch.no_grad():
#         for batch_x, batch_y in dataloader:
#             outputs = model(batch_x.view(-1, input_size*2))
#             loss = PULoss(outputs)
#             total_loss += loss.item() * batch_x.size(0)
#             preds = (outputs > 0.5).int()
#             total_correct += torch.sum(preds == batch_y).item()
#             total_samples += batch_x.size(0)
#     accuracy = total_correct / total_samples
#     return total_loss / total_samples, accuracy

# # Train the model
# n_epochs = 50
# best_loss = float('inf')

# for epoch in range(n_epochs):
#     model.train()
#     total_loss = 0.0
#     for batch_x, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_x.view(-1, input_size*2))
#         loss = PULoss(outputs)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * batch_x.size(0)
#     scheduler.step()
#     train_loss = total_loss / len(train_dataset)
#     val_loss, val_acc = evaluate(model, val_loader)
#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#     if val_loss < best_loss:
#         torch.save(model.state_dict(), "best_model.pth")
#         best_loss = val_loss



###################

import pandas as pd
import numpy as np

# p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
def eval_ranking(predp,outp):
    outf = open(outp,'w')
    # p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
    predresult = pd.read_pickle(predp)
    predresult['protein1'] = predresult['pairs'].map(lambda x:int(x[0]))
    predresult['protein2'] = predresult['pairs'].map(lambda x:int(x[1]))
    aa = predresult.sort_values(by=['protein1','protein2'])
    # print('>>>>>>\n',aa[:5])
    unique_proteins = np.unique(aa[["protein1", "protein2"]].values)
    a1_list=[]
    a10_list=[]
    n=0
    print('total={}'.format(len(unique_proteins)))
    for protein in unique_proteins:
        n+=1
        if n % (len(unique_proteins)//10)==0:
            print('>>{}/{}'.format(n,len(unique_proteins)))
        allinters = aa.query("protein1==@protein | protein2==@protein")
        allinters = allinters.sort_values(by=['preds'],ascending=False)
        # negvals = allinters.query('labels==0.0')['preds']
        posvals = allinters.query('labels==1.0')['preds']
        # print('posvals=\n',posvals)
        if len(posvals) >=1:
            # print('>>>>>all top10\n',allinters.sort_values(by=['preds'],ascending=False)[:10])
            at1 = allinters['labels'].values[0]
            
            if len(posvals) >=10:
                top10 = allinters[:10]
                at10 = sum(top10['labels'])
                a1_list.append(at1)
                a10_list.append(at10)
                print('{} \t @1={} @10={}'.format(protein,at1,at10),file=outf)
            else:
                print('{} \t @1={} @10=None'.format(protein,at1),file=outf,flush=True)

    print('a1_list\ntotal={},hit@1={}'.format(len(a1_list),sum(a1_list)/len(a1_list)),file=outf)
    print('a10_list\ntotal={},hit@10={}'.format(len(a10_list),sum(a10_list)/(10*len(a10_list))),file=outf)
            



'''    >>>>>>
                pairs  labels     preds  pr1   pr2
7251    (1.0, 571.0)     0.0  0.000008    1   571
17366  (1.0, 1581.0)     0.0  0.000673    1  1581
5313   (1.0, 2835.0)     0.0  0.000626    1  2835
7530   (1.0, 3740.0)     0.0  0.027729    1  3740'''
    
    

# p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
# outp = '/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/pu_model/eval@10.txt'

p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_onef_dot_64.pkl'
outp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_E@10_rk0408_40_onef_dot_64.txt'
eval_ranking(p,outp)

# outp='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_E@10_rk0408_40_twof_dot_64.txt'
# p='/ibex/scratch/projects/c2014/kexin/ppiproject/ccembed/scr/model/data/yeastdata/predictions_rk0408_40_twof_dot_64.pkl'
# eval_ranking(p,outp)







