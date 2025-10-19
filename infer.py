import json, os, numpy as np, torch, itertools, pandas as pd
from model import GATLinkPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS = os.getenv("ARTIFACTS", "artifacts")

x = torch.tensor(np.load(f"{ARTIFACTS}/x.npy"), dtype=torch.float32).to(DEVICE)
edge_index = torch.tensor(np.load(f"{ARTIFACTS}/edge_index.npy"), dtype=torch.long).to(DEVICE)
with open(f"{ARTIFACTS}/id_map.json") as f:
    id_map = json.load(f)
rev_id = {v:k for k,v in id_map.items()}

model = GATLinkPredictor(in_channels=x.size(1)).to(DEVICE)
model.load_state_dict(torch.load(f"{ARTIFACTS}/best_model.pt", map_location=DEVICE))
model.eval()

# Build candidate pairs not in edge_index:
edges = set(map(tuple, edge_index.cpu().t().numpy()))
N = x.size(0)
cands = []
for i, j in itertools.combinations(range(N), 2):
    if (i,j) not in edges and (j,i) not in edges:
        cands.append([i,j])
edge_label_index = torch.tensor(cands, dtype=torch.long).t().to(DEVICE)

with torch.no_grad():
    z = model.encode(x, edge_index)
    logits = model.decode(z, edge_label_index).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))

topk = 200
rows = []
for (i, j), p in zip(zip(*edge_label_index.cpu().numpy()), probs):
    rows.append((rev_id[i], rev_id[j], float(p)))
rows.sort(key=lambda r: r[2], reverse=True)
pd.DataFrame(rows[:topk], columns=["drug_a","drug_b","score"]).to_csv(f"{ARTIFACTS}/topk_predictions.csv", index=False)
print("Saved top candidates to artifacts/topk_predictions.csv")
