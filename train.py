import os, json, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import from_scipy_sparse_matrix
from model import GATLinkPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARTIFACTS = os.getenv("ARTIFACTS", "artifacts")

x = torch.tensor(np.load(f"{ARTIFACTS}/x.npy"), dtype=torch.float32)
edge_index = torch.tensor(np.load(f"{ARTIFACTS}/edge_index.npy"), dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

splitter = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True,
                           add_negative_train_samples=True)
train_data, val_data, test_data = splitter(data)

model = GATLinkPredictor(in_channels=data.num_node_features).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

def batchify(d):
    return (d.x.to(DEVICE), d.edge_index.to(DEVICE),
            d.edge_label_index.to(DEVICE), d.edge_label.to(DEVICE).float())

best_val = -1; best_path = f"{ARTIFACTS}/best_model.pt"
for epoch in range(1, 101):
    model.train()
    x, ei, eli, y = batchify(train_data)
    opt.zero_grad()
    pred = model(x, ei, eli)
    loss = loss_fn(pred, y)
    loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        def eval_split(d):
            x, ei, eli, y = batchify(d)
            logits = model(x, ei, eli).detach().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            return (roc_auc_score(y.cpu(), probs),
                    average_precision_score(y.cpu(), probs))
        val_auc, val_ap = eval_split(val_data)
    if val_auc > best_val:
        best_val = val_auc
        torch.save(model.state_dict(), best_path)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_auc={val_auc:.4f} | val_ap={val_ap:.4f}")

# final test
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()
with torch.no_grad():
    x, ei, eli, y = batchify(test_data)
    logits = model(x, ei, eli).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    test_auc = roc_auc_score(y.cpu(), probs)
    test_ap  = average_precision_score(y.cpu(), probs)
json.dump({"test_auc": float(test_auc), "test_auprc": float(test_ap)}, open(f"{ARTIFACTS}/metrics.json","w"))
print("TEST:", {"auc": test_auc, "auprc": test_ap})
