# Architecture

`mermaid
flowchart LR
  A[CSV Dataset data/raw/churn.csv] --> B[Load & Validate]
  B --> C[Preprocess: split, encode, scale]
  C --> D[Train Models]
  D --> E[Evaluate: ROC-AUC, F1, CM]
  E --> F[Persist Models models/*.joblib]
  E --> G[Streamlit Dashboard]
  F --> G
