import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class ProteinInteractionDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform ## normalization or something
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]
        if self.transform:
            pair = self.transform(pair)
        return pair, label

class ProteinInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ProteinInteractionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

def PULoss(outputs):
    pos = outputs[labels == 1]
    neg = outputs[labels == 0]
    pos_loss = torch.mean(torch.log(pos))
    neg_loss = torch.mean(torch.log(1-neg))
    loss = -(pos_loss + neg_loss)
    return loss

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            outputs = model(batch_x.view(-1, input_size*2))
            loss = PULoss(outputs)
            total_loss += loss.item() * batch_x.size(0)
            predicted = torch.round(outputs)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
    loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    return loss, accuracy

# sample data
pairs = torch.randn(1000, 2, 1280)
labels = torch.randint(0, 2, (1000,))
labels[labels == 0] = -1

# split data into train, validation and test sets
train_pairs, train_labels = pairs[:600], labels[:600]
val_pairs, val_labels = pairs[600:800], labels[600:800]
test_pairs, test_labels = pairs[800:], labels[800:]

# create datasets and data loaders
train_dataset = ProteinInteractionDataset(train_pairs, train_labels)
val_dataset = ProteinInteractionDataset(val_pairs, val_labels)
test_dataset = ProteinInteractionDataset(test_pairs, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# create model, loss function and optimizer
input_size = 1280
hidden_size = 128
output_size = 1
model = ProteinInteractionModel(input_size*2, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# train the model
n_epochs = 50
best_loss = float("inf")
for epoch in range(n_epochs):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x.view(-1, input_size*2))
        loss = PULoss(outputs)
        total_loss += loss.item() * batch_x.size(0)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    # evaluate on validation set
    val_loss, val_accuracy = evaluate(model, val_loader)
    print("Epoch {} | Train Loss: {:.4f} | Val Loss: {:.4f} | Val Accuracy: {:.4f}".format(epoch+1, total_loss/len(train_loader.dataset), val_loss, val_accuracy))
    
    # save the best model
    if val_loss < best_loss:
        torch.save(model.state_dict(), "best_model.pth")
        best_loss = val_loss

# evaluate the best model on test set
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_accuracy = evaluate(model, test_loader)
print("Test Loss: {:.4f} | Test Accuracy: {:.4f}".format(test_loss, test_accuracy))
