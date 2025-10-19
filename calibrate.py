#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temperature scaling for GAT DDI link predictor.
Fits a single temperature T on the validation split and saves artifacts/calibration.json.
"""
import os, json, numpy as np, torch

from model import GATLinkPredictor

ART = os.getenv("ARTIFACTS", "artifacts")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_for_calibration():
    x    = torch.tensor(np.load(f"{ART}/x.npy"), dtype=torch.float32).to(DEVICE)
    tei  = torch.tensor(np.load(f"{ART}/train_edge_index.npy"), dtype=torch.long).to(DEVICE)
    veli = torch.tensor(np.load(f"{ART}/val_edge_label_index.npy"), dtype=torch.long).to(DEVICE)
    vy   = torch.tensor(np.load(f"{ART}/val_edge_label.npy"), dtype=torch.float32).to(DEVICE)
    m = GATLinkPredictor(in_channels=x.size(1)).to(DEVICE)
    m.load_state_dict(torch.load(f"{ART}/best_model.pt", map_location=DEVICE))
    m.eval()
    with torch.no_grad():
        z = m.encode(x, tei)
        logits = m.decode(z, veli)
    return logits.detach(), vy

def main():
    logits, y = load_for_calibration()
    T = torch.nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        p = torch.sigmoid(logits / T)
        eps = 1e-8
        loss = -(y*torch.log(p+eps) + (1-y)*torch.log(1-p+eps)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    Tval = float(T.detach().cpu())
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "calibration.json"), "w") as f:
        json.dump({"temperature": Tval}, f, indent=2)
    print(f"[OK] Saved {os.path.join(ART, 'calibration.json')} with T={Tval:.6f}")

if __name__ == "__main__":
    main()
