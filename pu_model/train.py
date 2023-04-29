import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import twoProteinInteractionModel, oneProteinInteractionModel

# Define the positive-unlabeled learning loss function
def PULoss(outputs, alpha=0.5, beta=0.5):
    pos = outputs[labels == 1]
    neg = outputs[labels == 0]
    unlabel = outputs[labels == 0.5]
    # Calculate the loss for positive examples
    loss_positive = -alpha * torch.mean(torch.log(pos))

    # Calculate the loss for negative examples
    loss_negative = -beta * torch.mean(torch.log(1-neg))

    # Calculate the loss for unlabel examples
    loss_unlabel = 
    # Total loss
    loss = loss_positive + loss_negative

    return loss



def train(modeltype,epochs,input_size,hidden_size,output_size,train_loader,val_loader):
    # create model, loss function and optimizer
    if modeltype=='1':
        model = oneProteinInteractionModel(input_size, hidden_size, output_size)
    elif modeltype =='2':
        model = twoProteinInteractionModel(input_size, hidden_size, output_size)  

    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    n_epochs = epochs
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
    return model, bestmodelp

# train with model 

model, best_modelp = train('1',50,1280,128,1280,train_loader=train_loader,val_loader=val_loader)

# evaluate the best model on test set
model.load_state_dict(torch.load("best_model.pth"))
test_loss, test_accuracy = evaluate(model, test_loader)
print("Test Loss: {:.4f} | Test Accuracy: {:.4f}".format(test_loss, test_accuracy))




