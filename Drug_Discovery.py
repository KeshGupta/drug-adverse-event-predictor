import pandas as pd
import numpy as np
import requests 
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

# Set variables for training the testing functions    
CONFIDENCE_THRESHOLD = 0.3
EPOCHS = 10

# MAIN FUNCTION

# Define the main function, runs the other functions and uses pickle to save the model weights and maps from drugs to reactions
def main(): 

    # Initialize lists and index variables to keep track of API calling and reactions lists
    pharma_list = []
    results = []
    i = 0

    # Since the FDA API has a limit at 1000 calls, call 500 at a time and randomize it to get a different dataset each time
    while len(results) < 10000:
        request = requests.get("https://api.fda.gov/drug/event.json", params = {
            "limit": 500,
            "skip": i*500
            } )

        data = request.json()
        if "results" not in data:
            print(f"API error at skip={i*500}:", data.get("error", "unknown error"))
            break
        results += random.sample(data["results"], k=min(250, len(data["results"])))

        i += 1

    trial = pd.DataFrame(results)

    # Initilize temporary variables and loop to create the dataset that will be fed into neural network
    for i in range(len(trial.index)):
        reactions = []
        pharma_dict = {}

        # Grab the product and indication from the messy dataset
        try:
            drug = trial.loc[i, "patient"]["drug"][0]["medicinalproduct"]
            condition = trial.loc[i, "patient"]["drug"][0]["drugindication"]
        except KeyError:
            continue
        # Grab the reactions from the dataset
        for symptom in trial.loc[i, "patient"]["reaction"]:
            if "reactionmeddrapt" in symptom:
                reactions.append(symptom["reactionmeddrapt"])

        if not reactions:
            continue

        pharma_dict["drug"] = drug
        pharma_dict["condition"] = condition
        pharma_dict["reactions"] = reactions

        # Create an interim dataset with the drugs, conditions, and reactions
        pharma_list.append(pharma_dict)


    # Create a dataframe which will be cleaned 
    new_data = pd.DataFrame(pharma_list)

    new_data = cleaning(new_data)

    # Create a variable mapping the drugs to conditions which will be passed to inference.py to map the input drug to it's conditions
    drug_to_conditions = new_data.groupby("drug")["condition"].unique().apply(list).to_dict()

    # Pass inputs into preprocessing
    model, train_loader, test_loader, loss_fn, device, optimizer, scheduler, mlb, X_df, drug_ohe, condition_ohe = preprocessing(new_data)

    # Pass outputs from preprocessing into training the dataset
    train_model(model, train_loader, device, optimizer, loss_fn, scheduler)

    # Test inputs and assess accuracy
    model, mlb, columns = evaluate(model, test_loader, device, mlb, X_df)


   
    # save all the model weights, drugs, conditions, and a map from drugs to conditions to pass into inference.py
    torch.save(model.state_dict(), "Project_0: Drug Discovery/drug_model.pth")

    with open("Project_0: Drug Discovery/mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)
    with open("Project_0: Drug Discovery/columns.pkl", "wb") as f:
        pickle.dump(columns, f)
    with open("Project_0: Drug Discovery/drug.pkl", "wb") as f:
        pickle.dump(list(drug_ohe.columns), f)
    with open("Project_0: Drug Discovery/conditions.pkl", "wb") as f:
        pickle.dump(list(condition_ohe.columns), f)
    with open("Project_0: Drug Discovery/drug_to_conditions.pkl", "wb") as f:
        pickle.dump(drug_to_conditions, f)
    


# This functions cleans the data and returns new_data as a cleaned dataset ready to be passes into the network
def cleaning(new_data):

    # Filter label "product used for unknown indication" 
    new_data = new_data.loc[new_data["condition"] != "PRODUCT USED FOR UNKNOWN INDICATION"]

    # Filter data by count. Make the network stronger by only including drugs/conditions with multiple appearances
    new_data = new_data.groupby("drug").filter(lambda x: len(x) > 10)
    new_data = new_data.groupby("condition").filter(lambda x: len(x) > 8)


    # Separate working copy for counting individual label frequency
    exploded = new_data.explode("reactions")
    reaction_counts = exploded["reactions"].value_counts()
    keep_reactions = set(reaction_counts[reaction_counts > 9].index)

    # Filter each row's reaction list down to only frequent reactions
    new_data["reactions"] = new_data["reactions"].apply(
        lambda lst: [r for r in lst if r in keep_reactions])

    # Drop rows that ended up with no reactions left
    new_data = new_data[new_data["reactions"].apply(len) > 0]

    # Grab information about data like unique drugs, conditions, and reactions
    print("Total rows:", len(new_data))
    print("Unique drugs:", new_data["drug"].nunique())
    print("Unique conditions:", new_data["condition"].nunique())
    print("Unique reaction labels:", len(keep_reactions))

    return new_data
  

# Define neural network class and forward loop
# Each step is it's own line to make things easier to read
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

# This functions preprocesses the data, returning the mode, loader, loss, and tensor encodings of the data
def preprocessing(new_data):

  
    TEST_SIZE = 0.4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # OHE the drug and conditions columns. OHE explodes into multiple columns and creates vectors 
    # to represent where the label is stored
    drug_ohe = pd.get_dummies(new_data["drug"])        
    condition_ohe = pd.get_dummies(new_data["condition"]) 

    # Concatenate side-by-side, this means each row is now a 136-dimensional input vector
    X_df = pd.concat([drug_ohe, condition_ohe], axis=1)  

    # reactions column holds Python lists. Use MLB to turn then into vectors
    # difference between MLB and OHE is that MLB allows for multiple 1's in the vector space
    mlb = MultiLabelBinarizer()
    Y_arr = mlb.fit_transform(new_data["reactions"]) 

   
    # COnvert to pytorch tensors, inputs are floats
    # Labels must also be float32 because BCEWithLogitsLoss only takes that format 
    X = torch.tensor(X_df.values, dtype=torch.float32)
    y = torch.tensor(Y_arr, dtype=torch.float32)

    # Use SK for train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    print("X_train shape:", X_train.shape)  # e.g. (1183, 136)
    print("y_train shape:", y_train.shape)  # e.g. (1183, 204)
    print("Classes (reactions):", len(mlb.classes_))

    # Change loss variable to how rare the reaction. Getting rare reactions false causes loss to increase more, punishing the model
    pos_counts = y_train.sum(dim=0)                         
    neg_counts = len(y_train) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1e-6)).to(device)  # ~20-50x weight on positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Create model with a scheduler to reduce learning rate after plateuing 
    model = DrugReactions(input_size=X_df.shape[1], output_size=len(mlb.classes_)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


    # Setup dataloader, turn shuffle on
    train_dataset = TensorDataset(X_train, y_train)     
    test_dataset  = TensorDataset(X_test, y_test)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset,  batch_size=32)

    return model, train_loader, test_loader, loss_fn, device, optimizer, scheduler, mlb, X_df, drug_ohe, condition_ohe

# Train the model using the model, optimizer, loss, scheduler defined in preprocessing
def train_model(model, train_loader, device, optimizer, loss_fn, scheduler):

    # Familiar architechture for epoch loop to train data
    for epoch in range(EPOCHS):
        model.train()                                
        total_loss = 0
        train_correct = 0
        train_samples = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()                       
            predictions = model(batch_X)               
            loss = loss_fn(predictions, batch_y)            
            loss.backward()                           
            optimizer.step()                      

            total_loss += loss.item()

            # Extra steps to print training accuracy, similar to in eval()
            with torch.no_grad():
                probs = torch.sigmoid(predictions)
                top2_probs, top2_idx = torch.topk(probs, k=2, dim=1) # Use confidence threshold to choose top 2 options
                keep = top2_probs > CONFIDENCE_THRESHOLD # Only take second options if it's greater than confidence threshold
                keep[:, 0] = True
                pred_labels = torch.zeros_like(probs)
                for j in range(2):
                    rows = keep[:, j].nonzero(as_tuple=True)[0]
                    pred_labels[rows, top2_idx[rows, j]] = 1.0

                # Algorithm counts prediction as correct if it got ONE reactions right
                train_correct += ((pred_labels * batch_y).sum(dim=1) > 0).sum().item() 
                train_samples += batch_y.size(0)

        train_acc = train_correct / train_samples
        scheduler.step(total_loss / len(train_loader)) # Update learning rate with scheduler
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/len(train_loader):.4f} - train acc: {train_acc:.4f}")

# Evaluate the model with evaluate()
def evaluate(model, test_loader, device, mlb, X_df):

    model.eval()                                    

    # similar loop as training 
    with torch.no_grad():
        correct = 0
        samples = 0
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions  = model(batch_X)
            probs = torch.sigmoid(predictions)      
            top2_probs, top2_idx = torch.topk(probs, k=2, dim=1)
            keep = top2_probs > CONFIDENCE_THRESHOLD
            keep[:, 0] = True # Convert tensor into True/False
            pred_labels  = torch.zeros_like(probs)
            for j in range(2):
                rows = keep[:, j].nonzero(as_tuple=True)[0] # COnvert Tensor to 0,1
                pred_labels[rows, top2_idx[rows, j]] = 1.0

            # Count as correct if one reactions is correct
            correct += ((pred_labels * batch_y).sum(dim=1) > 0).sum().item()
            samples += batch_y.size(0)

    print(f"Test accuracy: {correct/samples:.4f}")

    return model, mlb, list(X_df.columns) # Return the model and mlb

if __name__ == "__main__":
    main()
    