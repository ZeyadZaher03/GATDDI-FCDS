#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess top-k DDI predictions:
- Join DrugBank IDs with drug names from data/drugs.csv
- (Optional) Add attention-based influential neighbors from the first GAT layer
Outputs artifacts/topk_enriched.csv
"""
import os, json, argparse, pandas as pd, numpy as np, torch

from model import GATLinkPredictor

ART = os.getenv("ARTIFACTS", "artifacts")
DATA_DIR = os.getenv("DATA_DIR", "data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_maps():
    with open(os.path.join(ART, "id_map.json"), "r") as f:
        id_map = json.load(f)
    rev_id = {v: k for k, v in id_map.items()}
    return id_map, rev_id

def load_drugs():
    path = os.path.join(DATA_DIR, "drugs.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("data/drugs.csv not found. Run convert_drugbank_csv.py first.")
    df = pd.read_csv(path)
    return df[["drug_id", "name"]].drop_duplicates(subset=["drug_id"])

def enrich_names(df_pred, df_drugs):
    return (df_pred
            .merge(df_drugs.rename(columns={"drug_id":"drug_a","name":"name_a"}), on="drug_a", how="left")
            .merge(df_drugs.rename(columns={"drug_id":"drug_b","name":"name_b"}), on="drug_b", how="left"))

def compute_attention_neighbors(neighbors_k=3, use_train_graph=True):
    # Load artifacts
    x = torch.tensor(np.load(os.path.join(ART, "x.npy")), dtype=torch.float32).to(DEVICE)
    # Choose graph for encoding and attention extraction
    ei_path = os.path.join(ART, "train_edge_index.npy") if use_train_graph else os.path.join(ART, "edge_index.npy")
    if not os.path.exists(ei_path) and use_train_graph:
        # fallback
        ei_path = os.path.join(ART, "edge_index.npy")
    edge_index = torch.tensor(np.load(ei_path), dtype=torch.long).to(DEVICE)

    # Build model and get attention on first GAT layer
    m = GATLinkPredictor(in_channels=x.size(1)).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(ART, "best_model.pt"), map_location=DEVICE))
    m.eval()

    with torch.no_grad():
        # Access first conv for attention
        conv0 = m.convs[0]
        out, (ei, alpha) = conv0(x, edge_index, return_attention_weights=True)
        # alpha: shape (E, heads) or (E,)
        if alpha.dim() == 2:
            alpha = alpha.mean(dim=1)  # average heads
        alpha = alpha.detach().cpu().numpy()
        ei = ei.detach().cpu().numpy()  # shape (2, E)

    # Build top-K neighbor list per source node based on attention
    src, dst = ei[0], ei[1]
    from collections import defaultdict
    neigh = defaultdict(list)  # node -> list[(alpha, neighbor)]
    for a, i, j in zip(alpha, src, dst):
        neigh[int(i)].append((float(a), int(j)))

    # Keep top-k per node
    topk_map = {}
    for i, lst in neigh.items():
        lst.sort(key=lambda t: t[0], reverse=True)
        topk_map[i] = lst[:neighbors_k]

    return topk_map

def format_neighbors(idx, topk_map, rev_id, df_drugs):
    if idx not in topk_map:
        return ""
    rows = []
    for a, j in topk_map[idx]:
        dbid = rev_id.get(j, f"IDX{j}")
        name_row = df_drugs.loc[df_drugs["drug_id"] == dbid, "name"]
        nm = name_row.iloc[0] if len(name_row) else ""
        rows.append(f"{dbid}:{nm} ({a:.3f})")
    return "; ".join(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbors", type=int, default=3, help="Top attention neighbors to include per node")
    ap.add_argument("--use-train-graph", action="store_true",
                    help="Use train_edge_index.npy for attention; fallback to edge_index.npy if missing")
    args = ap.parse_args()

    preds_path = os.path.join(ART, "topk_predictions.csv")
    if not os.path.exists(preds_path):
        raise FileNotFoundError("artifacts/topk_predictions.csv not found. Run infer.py first.")
    df_pred = pd.read_csv(preds_path)

    df_drugs = load_drugs()
    id_map, rev_id = load_maps()

    # Enrich with names
    df_en = enrich_names(df_pred, df_drugs)

    # Attention neighbors
    try:
        topk_map = compute_attention_neighbors(neighbors_k=args.neighbors, use_train_graph=args.use_train_graph)
        # Map DrugBank IDs -> indices
        idx_map = {dbid: idx for dbid, idx in id_map.items()}
        neigh_a, neigh_b = [], []
        for _, row in df_en.iterrows():
            ia = idx_map.get(row["drug_a"])
            ib = idx_map.get(row["drug_b"])
            neigh_a.append(format_neighbors(ia, topk_map, rev_id, df_drugs) if ia is not None else "")
            neigh_b.append(format_neighbors(ib, topk_map, rev_id, df_drugs) if ib is not None else "")
        df_en["influential_neighbors_a"] = neigh_a
        df_en["influential_neighbors_b"] = neigh_b
    except Exception as e:
        # Non-fatal: still write names + scores
        print(f"[warn] attention extraction failed ({e}); writing without neighbor explanations")

    out_path = os.path.join(ART, "topk_enriched.csv")
    df_en.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} (rows={len(df_en)})")

if __name__ == "__main__":
    main()
