
GATDDI-FCDS — Drug–Drug Interaction Prediction with Graph Attention Networks
============================================================================

A research prototype that predicts drug–drug interactions (DDIs) using a Graph Attention Network (GAT)
over a drug graph built from DrugBank-like CSV exports. It supports robust data conversion, fingerprint
feature generation, GAT training, and memory‑safe inference with rich logging.

--------------------------------------------------------------------------------

Project structure
-----------------
(If your scripts live under `src/`, just prefix commands with `python src/...`)

.
├─ build_graph.py              # Build edge_index from ddi.csv + id_map.json
├─ convert_drugbank_csv.py     # Raw CSV → data/drugs.csv + data/ddi.csv
├─ explain.py                  # (Optional) Inspect attention weights
├─ infer.py                    # Memory-safe inference (similarity / sample / all) + detailed logs
├─ model.py                    # GAT encoder + MLP link predictor
├─ prepare_features.py         # SMILES → ECFP4 features (x.npy) + id_map.json
├─ train.py                    # Split, train, eval; writes best_model.pt + metrics.json
├─ data/                       # Put your raw and processed CSVs here
├─ artifacts/                  # All generated artifacts end up here
└─ merged_output.csv           # (example output; safe to ignore or delete)

--------------------------------------------------------------------------------

Quickstart
----------

0) Environment (Python 3.10+)

Option A (pip; simplest):
    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    # CPU PyTorch example; use CUDA wheel if you have a GPU
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch_geometric
    pip install rdkit
    pip install pandas numpy scikit-learn tqdm psutil

Option B (conda; RDKit friendly):
    conda create -n gatddi python=3.10 -y
    conda activate gatddi
    conda install -c pytorch pytorch cpuonly -y
    conda install -c conda-forge rdkit -y
    pip install torch_geometric pandas numpy scikit-learn tqdm psutil

Notes:
- GPU users: install a CUDA-matching PyTorch first, then torch_geometric (see their install notes).
- RDKit wheels are available on PyPI; on some platforms, conda-forge installs are the smoothest.

--------------------------------------------------------------------------------

Input data format
-----------------
Place your raw CSV at `data/drugbank_raw.csv` (or set IN_CSV env var). Minimum required columns:

- drugbank_id       e.g., DB01234
- name
- smiles
- drug-interactions list or delimited string of DrugBank IDs (e.g., ["DB00001","DB00002"] or DB00001; DB00002)

--------------------------------------------------------------------------------

End-to-end pipeline
-------------------

1) Convert raw CSV → graph inputs
    python convert_drugbank_csv.py
    # ENV overrides (optional):
    #   IN_CSV=data/my_raw.csv   OUT_DIR=data

Writes:
- data/drugs.csv  → drug_id,name,smiles (unique)
- data/ddi.csv    → head_id,tail_id,label (positives, undirected & deduped)

What it does:
- Normalizes IDs (db0001 → DB0001)
- Robustly parses `drug-interactions` (JSON, Python-list, or ; , | delimited)
- Drops self-loops and edges to unknown drugs

2) SMILES → ECFP4 features (x.npy) + id map
    python prepare_features.py

Writes:
- artifacts/x.npy        shape (N, 1024) float32 (ECFP4 / Morgan, radius=2)
- artifacts/id_map.json  {"DB00001": 0, ...}

Notes:
- Missing/invalid SMILES get a zero vector (OK for a baseline; can enrich later).

3) Build the graph (edge_index)
    python build_graph.py

Writes:
- artifacts/edge_index.npy  shape (2, 2E) int64 (mirrored undirected edges)

4) Train the GAT link predictor
    python train.py

Writes:
- artifacts/best_model.pt
- artifacts/metrics.json   e.g., {"test_auc": 0.87, "test_auprc": 0.81}

What happens:
- RandomLinkSplit (train/val/test) with negative sampling
- GAT encoder + MLP edge scorer
- Tracked metrics: AUROC & AUPRC

5) Memory-safe inference with logging

Recommended (fast + scalable) — similarity mode:
    python infer.py --mode similarity --per_node 200 --topk 500 --log-level INFO

Other modes:
    # Random large subset
    python infer.py --mode sample --sample_pairs 3000000 --topk 500

    # Exhaustive (stream ALL unknown pairs; slow but safe for small N)
    python infer.py --mode all --chunk_pairs 200000 --topk 500

Writes:
- artifacts/topk_predictions.csv → drug_a, drug_b, score (probabilities)
- artifacts/infer_summary.json   → run summary + score stats
- artifacts/infer.log            → detailed progress (memory, throughput)

--------------------------------------------------------------------------------

Model: GAT encoder + MLP link scorer
------------------------------------

Node features:
- ECFP4 (1024-d) from SMILES

GAT (per layer, per head):
- Linear projection:     h'_i = W h_i
- Attention scores:      e_ij = LeakyReLU(a^T [h'_i || h'_j])
- Normalization:         α_ij = softmax_j(e_ij) over neighbors of i
- Aggregation:           z_i = σ( Σ_j α_ij W h'_j )
- Multi-head: concat outputs across H heads
- Depth: typically 2 layers (can go deeper with care)

Edge scoring:
- Pair feature: g_ij = [z_i || z_j] → MLP → logit → sigmoid → probability

Loss & metrics:
- Loss: BCEWithLogitsLoss
- Eval: AUROC + AUPRC (AUPRC is more meaningful under class imbalance)

--------------------------------------------------------------------------------

Outputs & where to look
-----------------------
- artifacts/x.npy             node features (N×1024)
- artifacts/id_map.json       node index map
- artifacts/edge_index.npy    graph connectivity
- artifacts/best_model.pt     trained weights
- artifacts/metrics.json      final test metrics
- artifacts/topk_predictions.csv  ranked unknown DDI candidates
- artifacts/infer_summary.json    inference summary + score distribution
- artifacts/infer.log             full logs (ETA, throughput, memory)

--------------------------------------------------------------------------------

Sanity checks
-------------
- data/ddi.csv: no duplicates, no self-loops
- artifacts/x.npy.shape == (N, 1024); most rows non-zero
- artifacts/edge_index.npy.shape == (2, 2E); indices in [0, N-1]
- artifacts/metrics.json shows non-trivial AUPRC
- artifacts/topk_predictions.csv contains only previously unknown pairs

--------------------------------------------------------------------------------

Useful flags (inference)
------------------------
Candidate selection:
- --mode {similarity,sample,all}
- --per_node 200          (similarity: candidates per node)
- --sample_pairs 3000000  (sample: total random pairs)
- --chunk_pairs 200000    (all: stream pairs per chunk)

Scoring:
- --topk 500              (keep global top-K predictions)
- --batch_edges 262144    (pairs scored per batch; lower to reduce memory)

Logging:
- --log-level {INFO,DEBUG}
- --log-file artifacts/infer.log
- --log-every-batches 10

The script prints memory usage (via psutil) and throughput periodically.

--------------------------------------------------------------------------------

Troubleshooting
---------------
- Process was “killed” during inference:
  Use --mode similarity or --mode sample, and/or lower --batch_edges.
  The script streams and keeps only a small global top-K heap.

- RDKit install issues:
  Prefer conda (`conda install -c conda-forge rdkit`) on some platforms.

- PyG install issues:
  Make sure PyTorch (CPU or CUDA) is installed first, then `pip install torch_geometric`.
  For CUDA, use the wheel URLs that match your Torch/CUDA.

- Low AUPRC:
  Many missing SMILES → add auxiliary features (ATC multi-hot, targets/enzymes).
  Consider hard-negative sampling (e.g., same ATC class) during training.

--------------------------------------------------------------------------------

Roadmap
-------
- ATC / target / enzyme features → concatenate to ECFP for better coverage
- Relation-typed edges (R-GAT) for mechanism-specific predictions
- Calibration (temperature scaling)
- Simple Streamlit demo for interactive scoring

--------------------------------------------------------------------------------

Data & licensing
----------------
This repo contains code only. Ensure your DrugBank/KEGG data usage complies with their terms/licenses.

--------------------------------------------------------------------------------
