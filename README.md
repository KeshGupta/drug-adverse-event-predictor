# drug-adverse-event-predictor
This network predicts adverse reactions given an input drug and medical condition. It's a multi label network using data from the FDA FAERS (adverse event reporting system) via the openFDA API

## What it does
Given a drug name and the condition it treats, the model returns the most probable adverse reactions with confidence scores, using a feedforward neural network with multi-label binary cross-entropy loss.

## How to run it
1. Install dependencies: `pip install pandas torch scikit-learn requests`
2. Run `Drug_Discovery.py` to pull data, train, and save the model
3. Run `Inference.py` to query the trained model interactively

## Key technical decisions
- One-hot encoding for drug/condition inputs (~136 dimensions) 
  rather than embeddings — appropriate at this vocabulary size
- BCEWithLogitsLoss with pos_weight to handle class imbalance 
  across reaction labels
- Multi-label binarization (not single-label) since a single 
  report can have multiple simultaneous reactions

## Known limitations
- Drug name normalization is incomplete — brand and generic names 
  for the same drug are treated as separate entities
- FAERS data is voluntarily reported; causal relationships between 
  drugs and reactions cannot be established from this data alone
- Evaluation uses a lenient "any overlap" accuracy metric; 
  precision/recall/F1 per label would be more rigorous
- Model generalizes only to drug/condition pairs seen in training

## Next steps
- Drug normalization via RxNorm API
- Replace one-hot with learned embeddings as vocabulary scales
- Add per-label precision/recall/F1 evaluation
- Incorporate molecular structure features (SMILES via PubChem)
