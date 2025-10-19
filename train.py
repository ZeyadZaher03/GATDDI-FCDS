#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GAT link predictor for DDI with better negatives and split artifacts saved for calibration.
"""
import os, json, random, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from model import GATLinkPredictor

ARTIFACTS = os.getenv("ARTIFACTS", "artifacts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(os.getenv("SEED", "42"))

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data():
    x = torch.tensor(np.load(f"{ARTIFACTS}/x.npy"), dtype=torch.float32)
    edge_index = torch.tensor(np.load(f"{ARTIFACTS}/edge_index.npy"), dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def eval_split(model, d, device):
    model.eval()
    with torch.no_grad():
        x, ei, eli, y = d.x.to(device), d.edge_index.to(device), d.edge_label_index.to(device), d.edge_label.to(device).float()
        logits = model(x, ei, eli).detach().cpu().numpy()
        probs  = 1 / (1 + np.exp(-logits))
        y_true = y.cpu().numpy()
        auc = roc_auc_score(y_true, probs)
        ap  = average_precision_score(y_true, probs)
        return auc, ap

def main():
    set_seed(SEED)
    os.makedirs(ARTIFACTS, exist_ok=True)
    data = load_data()

    splitter = RandomLinkSplit(
        num_val=0.1, num_test=0.1, is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=5.0  # harder task (more negatives)
    )
    train_data, val_data, test_data = splitter(data)

    # Save split artifacts for calibration / analysis
    np.save(f"{ARTIFACTS}/train_edge_index.npy", train_data.edge_index.numpy())
    np.save(f"{ARTIFACTS}/val_edge_label_index.npy",  val_data.edge_label_index.numpy())
    np.save(f"{ARTIFACTS}/val_edge_label.npy",        val_data.edge_label.numpy())
    np.save(f"{ARTIFACTS}/test_edge_label_index.npy", test_data.edge_label_index.numpy())
    np.save(f"{ARTIFACTS}/test_edge_label.npy",       test_data.edge_label.numpy())

    model = GATLinkPredictor(in_channels=data.num_node_features).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_path = f"{ARTIFACTS}/best_model.pt"

    def batchify(d):
        return (d.x.to(DEVICE), d.edge_index.to(DEVICE),
                d.edge_label_index.to(DEVICE), d.edge_label.to(DEVICE).float())

    for epoch in range(1, 121):
        model.train()
        x, ei, eli, y = batchify(train_data)
        opt.zero_grad()
        pred = model(x, ei, eli)
        loss = loss_fn(pred, y)
        loss.backward(); opt.step()

        # Evaluate
        val_auc, val_ap = eval_split(model, val_data, DEVICE)

        if val_auc > best_val:
            best_val = val_auc
            torch.save(model.state_dict(), best_path)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_auc={val_auc:.4f} | val_auprc={val_ap:.4f}")

    # Final test evaluation with best model
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_auc, test_ap = eval_split(model, test_data, DEVICE)
    metrics = {"test_auc": float(test_auc), "test_auprc": float(test_ap)}
    json.dump(metrics, open(f"{ARTIFACTS}/metrics.json","w"))
    print("[TEST]", metrics)

if __name__ == "__main__":
    main()
