import json, os, numpy as np, pandas as pd, torch
from torch_geometric.data import Data

DATA_DIR = os.getenv("DATA_DIR", "data")
ARTIFACTS = os.getenv("ARTIFACTS", "artifacts")

X = np.load(f"{ARTIFACTS}/x.npy")
with open(f"{ARTIFACTS}/id_map.json") as f:
    id_map = json.load(f)

ddi = pd.read_csv(f"{DATA_DIR}/ddi.csv")  # head_id, tail_id, label (1)
# Canonicalize pairs and keep valid nodes:
def to_idx(row):
    if row.head_id in id_map and row.tail_id in id_map:
        a, b = id_map[row.head_id], id_map[row.tail_id]
        return (min(a,b), max(a,b))
    return None
pairs = ddi.apply(to_idx, axis=1).dropna().tolist()
pairs = list(set(pairs))  # dedupe
edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected

x = torch.tensor(X, dtype=torch.float32)
data = Data(x=x, edge_index=edge_index)
np.save(f"{ARTIFACTS}/edge_index.npy", edge_index.numpy())
print(data)
