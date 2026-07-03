import pandas as pd
import torch
import torch.nn as nn
import pickle
import sys


# Reload all data we saved like model weights and columns and a map from drugs to conditions
with open("Project_0: Drug Discovery/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("Project_0: Drug Discovery/columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("Project_0: Drug Discovery/drug.pkl", "rb") as f:
    drug = pickle.load(f)

with open("Project_0: Drug Discovery/conditions.pkl", "rb") as f:
    conditions = pickle.load(f)

with open("Project_0: Drug Discovery/drug_to_conditions.pkl", "rb") as f:
    drug_to_conditions = pickle.load(f)

# Rebuild neural network architechture 
class DrugReactions(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.net(x)

model = DrugReactions(input_size=len(columns), output_size=len(mlb.classes_))

model.load_state_dict(torch.load("Project_0: Drug Discovery/drug_model.pth"))

# Print the available drugs in the drug list for the user to choose
print("Available drugs:", drug)

# Print the conditions the drug was used to treat
input_drug = input("Name a drug from the following: ")
if input_drug not in drug:
    print("try again, drug not in database")
    sys.exit()
print(drug_to_conditions[input_drug])
input_condition = input("What condition is your drug trying to treat: ")

if input_condition not in drug_to_conditions[input_drug]:
    print("try again, condition not in database")
    sys.exit()

# Rerun network with the same weights and architechture
X_input = pd.Series(0.0, index=columns)

X_input[input_drug] = 1       
X_input[input_condition] = 1

X_input = torch.tensor(X_input.values, dtype=torch.float32)
batch_X = X_input.unsqueeze(0)

CONFIDENCE_THRESHOLD = 0.4
model.eval()                                   

# Same loop as in eval()
with torch.no_grad():
    predictions  = model(batch_X)
    probs = torch.sigmoid(predictions)      
    top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
    keep = top2_probs > CONFIDENCE_THRESHOLD
    keep[:, 0] = True
    pred_labels  = torch.zeros_like(probs)
    for j in range(2):
        rows = keep[:, j].nonzero(as_tuple=True)[0]
        pred_labels[rows, top2_idx[rows, j]] = 1.0    

# Now print the reactions the network guesses as well as the accuracy it's guessing with 
indices = pred_labels[0].nonzero(as_tuple=True)[0]

for j in range(2):
    if keep[0, j]:
        reaction_name = mlb.classes_[top2_idx[0, j]]
        probability = top2_probs[0, j].item()
        print(f"{reaction_name}: {probability:.1%}")
        


    
