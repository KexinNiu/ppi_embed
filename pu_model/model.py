
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class twoProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(twoProteinInteractionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x0,x1 = x.chunk(2,axis=1)

        out0 = self.fc1(x0)
        out0 = self.relu(out0)
        out0 = self.fc2(out0)
        
        out1 = self.fc1(x1)
        out1 = self.relu(out1)
        out1 = self.fc2(out1)

        dotproduct = torch.sum(out0*out1,dim=-1)
        return dotproduct
    
class oneProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(oneProteinInteractionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0,x1 = x.chunk(2,axis=1)
        out0 = self.fc1(x0)
        out0 = self.relu(out0)
        out0 = self.fc2(out0)
        dotproduct = torch.sum(out0*x1,dim=-1)
        return dotproduct  