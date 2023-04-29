import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load protein interactions data from the STRING database
interactions = pd.read_csv("string_interactions.csv")

# Filter for high confidence interactions
high_confidence_threshold = 0.9
interactions = interactions[interactions["combined_score"] >= high_confidence_threshold]

# Get a list of unique proteins
unique_proteins = np.unique(interactions[["protein1", "protein2"]].values)

# Create a dictionary to map new index names to original protein names
index_to_protein = dict(enumerate(unique_proteins))
protein_to_index = {v: k for k, v in index_to_protein.items()}

# Replace protein names with new indices in the interactions dataframe
interactions["protein1"] = interactions["protein1"].apply(lambda x: protein_to_index[x])
interactions["protein2"] = interactions["protein2"].apply(lambda x: protein_to_index[x])

# Split proteins into 3 sets
proteins_train, proteins_val, proteins_test = np.split(np.random.permutation(unique_proteins), [int(0.6*len(unique_proteins)), int(0.8*len(unique_proteins))])

# Assign interactions to each set
interactions_train = interactions[(interactions["protein1"].isin(protein_to_index[x] for x in proteins_train)) & (interactions["protein2"].isin(protein_to_index[x] for x in proteins_train))]
interactions_val = interactions[(interactions["protein1"].isin(protein_to_index[x] for x in proteins_val)) & (interactions["protein2"].isin(protein_to_index[x] for x in proteins_val))]
interactions_test = interactions[(interactions["protein1"].isin(protein_to_index[x] for x in proteins_test)) & (interactions["protein2"].isin(protein_to_index[x] for x in proteins_test))]

# Save the split datasets to separate files
interactions_train.to_csv("interactions_train.csv", index=False)
interactions_val.to_csv("interactions_val.csv", index=False)
interactions_test.to_csv("interactions_test.csv", index=False)

# Create positive-unlabeled (PU) datasets for PyTorch
from torch.utils.data import Dataset

class PUDataset(Dataset):
    def __init__(self, pos_data, unlab_data):
        self.pos_data = pos_data
        self.unlab_data = unlab_data
    
    def __len__(self):
        return len(self.pos_data) + len(self.unlab_data)
    
    def __getitem__(self, idx):
        if idx < len(self.pos_data):
            x, y = self.pos_data.iloc[idx, :]["protein1"], self.pos_data.iloc[idx, :]["protein2"]
            label = 1
        else:
            idx -= len(self.pos_data)
            x, y = self.unlab_data.iloc[idx, :]["protein1"], self.unlab_data.iloc[idx, :]["protein2"]
            label = 0
        return x, y, label

# Create a dictionary to map new index names back to original protein names
index_to_protein = {v: k for k, v in protein_to_index.items()}

# Save the index to protein dictionary to a file
with open("index_to_protein.txt", "w") as f:
    for index, protein in index_to_protein.items():
        f.write(f"{index}\t{protein}\n")

# Load the split
# Load the index to protein dictionary from the file
index_to_protein = {}
with open("index_to_protein.txt", "r") as f:
    for line in f:
        index, protein = line.strip().split("\t")
        index_to_protein[int(index)] = protein

# Load the split datasets from the files
interactions_train = pd.read_csv("interactions_train.csv")
interactions_val = pd.read_csv("interactions_val.csv")
interactions_test = pd.read_csv("interactions_test.csv")

# Replace index names with original protein names in the interactions dataframes
interactions_train["protein1"] = interactions_train["protein1"].apply(lambda x: index_to_protein[x])
interactions_train["protein2"] = interactions_train["protein2"].apply(lambda x: index_to_protein[x])

interactions_val["protein1"] = interactions_val["protein1"].apply(lambda x: index_to_protein[x])
interactions_val["protein2"] = interactions_val["protein2"].apply(lambda x: index_to_protein[x])

interactions_test["protein1"] = interactions_test["protein1"].apply(lambda x: index_to_protein[x])
interactions_test["protein2"] = interactions_test["protein2"].apply(lambda x: index_to_protein[x])

# Create positive-unlabeled (PU) datasets for PyTorch
train_dataset = PUDataset(interactions_train, interactions_val.append(interactions_test))
val_dataset = PUDataset(interactions_val, interactions_train.append(interactions_test))
test_dataset = PUDataset(interactions_test, interactions_train.append(interactions_val))
