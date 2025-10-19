# src/infer.py
import os, json, argparse, heapq, time, math, numpy as np, torch, logging, statistics
from collections import defaultdict
from datetime import datetime

try:
    import psutil
except Exception:
    psutil = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from model import GATLinkPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ART = os.getenv("ARTIFACTS", "artifacts")

# ----------------------------- Logging helpers -----------------------------

def setup_logger(level: str = "INFO", logfile: str | None = None):
    logger = logging.getLogger("infer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(sh)
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def mem_str():
    if psutil is None:
        return "mem: n/a"
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    gb = rss / (1024**3)
    return f"mem: {gb:.2f} GB"

class Timer:
    def __init__(self, logger, label):
        self.logger = logger
        self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
        self.logger.debug(f"[start] {self.label}")
        return self
    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        self.logger.info(f"[done]  {self.label} in {dt:.2f}s ({mem_str()})")

def fmtn(x):  # nice thousands
    return f"{x:,}"

# ----------------------------- IO / Model ----------------------------------

def load_artifacts(logger):
    with Timer(logger, "load x.npy"):
        x = torch.tensor(np.load(f"{ART}/x.npy"), dtype=torch.float32)
    with Timer(logger, "load edge_index.npy"):
        edge_index = torch.tensor(np.load(f"{ART}/edge_index.npy"), dtype=torch.long)
    with Timer(logger, "load id_map.json"):
        with open(f"{ART}/id_map.json") as f:
            id_map = json.load(f)
    rev_id = {v: k for k, v in id_map.items()}
    N = x.size(0)
    logger.info(f"nodes={fmtn(N)}, feat_dim={x.size(1)}, edges(dir)={fmtn(edge_index.size(1))} ({mem_str()})")
    return x, edge_index, rev_id

def count_params(m):
    return sum(p.numel() for p in m.parameters())

def load_model(logger, in_dim):
    with Timer(logger, "load model"):
        m = GATLinkPredictor(in_channels=in_dim).to(DEVICE)
        m.load_state_dict(torch.load(f"{ART}/best_model.pt", map_location=DEVICE))
        m.eval()
    logger.info(f"model params={fmtn(count_params(m))}, device={DEVICE}")
    return m

def encode_nodes(logger, model, x, edge_index):
    with Timer(logger, "encode nodes (GAT)"):
        with torch.no_grad():
            z = model.encode(x.to(DEVICE), edge_index.to(DEVICE))
    logger.info(f"embedding z: shape={tuple(z.shape)} ({mem_str()})")
    return z

def known_undirected_pairs(logger, edge_index):
    ei = edge_index.cpu().numpy()
    known = set()
    # edge_index is directed (mirrored), we condense to undirected
    for u, v in zip(ei[0], ei[1]):
        if u == v:
            continue
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        known.add((a, b))
    logger.info(f"known undirected edges≈{fmtn(len(known))}")
    return known

# ------------------------- Candidate generators -----------------------------

def iter_all_unknown_pairs(logger, N, known, chunk_pairs=200_000, log_every=1000):
    """Stream ALL unknown (i<j) pairs; yield chunks as tensors."""
    logger.info(f"[all] streaming all unknown pairs with chunk={fmtn(chunk_pairs)}")
    u = np.empty(chunk_pairs, dtype=np.int64)
    v = np.empty(chunk_pairs, dtype=np.int64)
    k = 0
    produced = 0
    for i in range(N):
        if i % log_every == 0 and i > 0:
            logger.debug(f"[all] i={fmtn(i)}/{fmtn(N)} ({mem_str()})")
        for j in range(i + 1, N):
            if (i, j) in known:
                continue
            u[k] = i; v[k] = j; k += 1
            if k == chunk_pairs:
                produced += k
                logger.debug(f"[all] yield chunk of {fmtn(k)}, produced={fmtn(produced)}")
                yield torch.from_numpy(u), torch.from_numpy(v)
                k = 0
    if k:
        produced += k
        logger.debug(f"[all] final chunk {fmtn(k)}, produced={fmtn(produced)}")
        yield torch.from_numpy(u[:k]), torch.from_numpy(v[:k])

def random_unknown_samples(logger, N, known, max_pairs, log_every=1_000_000):
    """Uniform sampling of unknown pairs until ~max_pairs collected."""
    logger.info(f"[sample] target={fmtn(max_pairs)} random pairs")
    rng = np.random.default_rng(42)
    batch = max(100_000, min(max_pairs, 1_000_000))
    out_u, out_v = [], []
    attempts = 0
    t0 = time.perf_counter()
    while len(out_u) < max_pairs:
        i = rng.integers(0, N, size=batch, endpoint=False, dtype=np.int64)
        j = rng.integers(0, N, size=batch, endpoint=False, dtype=np.int64)
        a = np.minimum(i, j); b = np.maximum(i, j)
        mask = a < b
        a, b = a[mask], b[mask]
        # drop known
        keep_u, keep_v = [], []
        for uu, vv in zip(a, b):
            if (int(uu), int(vv)) not in known:
                keep_u.append(int(uu)); keep_v.append(int(vv))
        out_u.extend(keep_u); out_v.extend(keep_v)
        attempts += batch
        if attempts % log_every == 0:
            rate = attempts / (time.perf_counter() - t0 + 1e-9)
            logger.debug(f"[sample] attempts={fmtn(attempts)} kept={fmtn(len(out_u))} rate={rate:,.0f}/s ({mem_str()})")
        if len(out_u) >= max_pairs:
            break
    arr_u = torch.tensor(np.array(out_u[:max_pairs], dtype=np.int64))
    arr_v = torch.tensor(np.array(out_v[:max_pairs], dtype=np.int64))
    logger.info(f"[sample] kept={fmtn(arr_u.numel())} (keep ratio≈{arr_u.numel()/(attempts+1e-9):.4f})")
    return arr_u, arr_v

def topk_similar_candidates(logger, Xbin, known, per_node=200, block=128):
    """
    Preselect up to 'per_node' nearest neighbors per node via Tanimoto,
    dedupe, and return candidates as (u,v) with u<v and unknown in graph.
    """
    N, F = Xbin.shape
    logger.info(f"[similarity] per_node={per_node}, block={block}, N={fmtn(N)}, F={F}")
    pop = Xbin.sum(axis=1)  # (N,)
    cand_u, cand_v = [], []
    total_searched = 0
    blocks = range(0, N, block)
    iterator = blocks if tqdm is None else tqdm(blocks, desc="[similarity] blocks")
    for start in iterator:
        end = min(N, start + block)
        A = Xbin[start:end]                   # (b,F)
        dot = A @ Xbin.T                      # (b,N)
        denom = (pop[start:end][:, None] + pop[None, :] - dot + 1e-8)
        tanimoto = dot / denom                # (b,N)
        for row, i in enumerate(range(start, end)):
            tanimoto[row, i] = -1.0  # remove self
        L = min(N, per_node * 2)
        idx = np.argpartition(-tanimoto, kth=min(L, tanimoto.shape[1]-1), axis=1)[:, :L]
        added_block = 0
        for row, i in enumerate(range(start, end)):
            js = idx[row]
            scores = tanimoto[row, js]
            order = np.argsort(-scores)
            added = 0
            for jj in js[order]:
                a, b = (i, int(jj)) if i < jj else (int(jj), i)
                if a == b: continue
                if (a, b) in known: continue
                cand_u.append(a); cand_v.append(b)
                added += 1
                if added >= per_node:
                    break
            added_block += added
        total_searched += (end - start) * N
        if tqdm is None:
            logger.debug(f"[similarity] block {start}:{end} added={fmtn(added_block)} ({mem_str()})")
    before = len(cand_u)
    pairs = list(set(zip(cand_u, cand_v)))
    uu, vv = zip(*pairs) if pairs else ([], [])
    arr_u = torch.tensor(np.array(uu, dtype=np.int64))
    arr_v = torch.tensor(np.array(vv, dtype=np.int64))
    logger.info(f"[similarity] raw={fmtn(before)} deduped={fmtn(arr_u.numel())} ({100*(1-arr_u.numel()/max(1,before)):.1f}% removed)")
    return arr_u, arr_v

# ------------------------------ Scoring -------------------------------------

def score_in_batches(logger, model, z, u, v, batch_edges=262_144, log_every_batches=10):
    """Yield (slice, probs) while logging throughput & memory occasionally."""
    m = u.numel()
    if m == 0:
        return
    t0 = time.perf_counter()
    batches = 0
    for s in range(0, m, batch_edges):
        e = min(m, s + batch_edges)
        edge_idx = torch.stack([u[s:e], v[s:e]], dim=0).to(DEVICE)
        with torch.no_grad():
            logits = model.decode(z, edge_idx).detach().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
        yield slice(s, e), probs
        batches += 1
        if batches % log_every_batches == 0:
            dt = time.perf_counter() - t0
            rate = (e) / (dt + 1e-9)
            logger.debug(f"[score] {fmtn(e)}/{fmtn(m)} pairs ({rate:,.0f} pairs/s, batch={fmtn(batch_edges)}) ({mem_str()})")

def write_topk_csv(logger, rev_id, top_pairs, out_path):
    import pandas as pd
    rows = [(rev_id[int(i)], rev_id[int(j)], float(s)) for s, i, j in top_pairs]
    rows.sort(key=lambda r: r[2], reverse=True)
    pd.DataFrame(rows, columns=["drug_a", "drug_b", "score"]).to_csv(out_path, index=False)
    logger.info(f"[OK] Saved {len(rows)} predictions → {out_path}")

def summarize_scores(logger, scores):
    if len(scores) == 0:
        return {}
    stats = {
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }
    logger.info(f"[scores] min={stats['min']:.4f} p50={stats['median']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f}")
    return stats

# --------------------------------- Main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["similarity", "sample", "all"], default="similarity",
                    help="Candidate generation strategy.")
    ap.add_argument("--topk", type=int, default=200, help="How many global top predictions to keep.")
    ap.add_argument("--per_node", type=int, default=200, help="similarity: candidates per node.")
    ap.add_argument("--sample_pairs", type=int, default=2_000_000, help="sample: #random pairs to score.")
    ap.add_argument("--chunk_pairs", type=int, default=200_000, help="all: stream this many pairs per chunk.")
    ap.add_argument("--batch_edges", type=int, default=262_144, help="scoring batch size (pairs).")
    ap.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    ap.add_argument("--log-file", default=os.path.join(ART, "infer.log"))
    ap.add_argument("--log-every-batches", type=int, default=10, help="log after this many scoring batches.")
    args = ap.parse_args()

    logger = setup_logger(args.log_level, args.log_file)
    logger.info(f"==== DDI-GAT INFER v1 ====")
    logger.info(f"cmd: mode={args.mode} topk={args.topk} per_node={args.per_node} sample_pairs={fmtn(args.sample_pairs)} "
                f"chunk_pairs={fmtn(args.chunk_pairs)} batch_edges={fmtn(args.batch_edges)} ({mem_str()})")

    x, edge_index, rev_id = load_artifacts(logger)
    model = load_model(logger, in_dim=x.size(1))
    z = encode_nodes(logger, model, x, edge_index)
    N = x.size(0)
    known = known_undirected_pairs(logger, edge_index)

    # Candidates
    if args.mode == "similarity":
        with Timer(logger, "candidate selection (similarity)"):
            Xbin = x.numpy().astype(np.float32)  # vectors are 0/1 floats
            u, v = topk_similar_candidates(logger, Xbin, known, per_node=args.per_node, block=128)
        logger.info(f"candidates={fmtn(u.numel())}")
        # Score and keep global top-K
        scores = np.empty(u.numel(), dtype=np.float32)
        k = 0
        with Timer(logger, f"scoring {fmtn(u.numel())} candidates"):
            for sl, probs in score_in_batches(logger, model, z, u, v,
                                              batch_edges=args.batch_edges,
                                              log_every_batches=args.log_every_batches):
                scores[sl] = probs
        K = min(args.topk, scores.size)
        top_idx = np.argpartition(-scores, K-1)[:K]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]
        top_pairs = [(float(scores[i]), int(u[i]), int(v[i])) for i in top_sorted]
        out = os.path.join(ART, "topk_predictions.csv")
        write_topk_csv(logger, rev_id, top_pairs, out)
        stats = summarize_scores(logger, scores[top_sorted])
        # summary file
        summary = {
            "mode": args.mode, "N": N, "known_edges": len(known),
            "candidates": int(u.numel()), "topk": int(K),
            "batch_edges": int(args.batch_edges), "device": DEVICE, "score_stats": stats,
        }
        json.dump(summary, open(os.path.join(ART, "infer_summary.json"), "w"), indent=2)
        logger.info(f"[OK] Summary → {os.path.join(ART,'infer_summary.json')}")

    elif args.mode == "sample":
        with Timer(logger, "candidate selection (random sample)"):
            u, v = random_unknown_samples(logger, N, known, max_pairs=args.sample_pairs)
        logger.info(f"candidates={fmtn(u.numel())}")
        scores = np.empty(u.numel(), dtype=np.float32)
        with Timer(logger, f"scoring {fmtn(u.numel())} candidates"):
            for sl, probs in score_in_batches(logger, model, z, u, v,
                                              batch_edges=args.batch_edges,
                                              log_every_batches=args.log_every_batches):
                scores[sl] = probs
        K = min(args.topk, scores.size)
        top_idx = np.argpartition(-scores, K-1)[:K]
        top_sorted = top_idx[np.argsort(-scores[top_idx])]
        top_pairs = [(float(scores[i]), int(u[i]), int(v[i])) for i in top_sorted]
        out = os.path.join(ART, "topk_predictions.csv")
        write_topk_csv(logger, rev_id, top_pairs, out)
        stats = summarize_scores(logger, scores[top_sorted])
        summary = {
            "mode": args.mode, "N": N, "known_edges": len(known),
            "candidates": int(u.numel()), "topk": int(K),
            "batch_edges": int(args.batch_edges), "device": DEVICE, "score_stats": stats,
        }
        json.dump(summary, open(os.path.join(ART, "infer_summary.json"), "w"), indent=2)
        logger.info(f"[OK] Summary → {os.path.join(ART,'infer_summary.json')}")

    else:  # args.mode == "all"
        logger.info("[all] exhaustive streaming — this can be slow; use only for modest N")
        top_heap = []  # (score, i, j)
        total_scored = 0
        with Timer(logger, "scoring ALL unknown pairs (streaming)"):
            for cu, cv in iter_all_unknown_pairs(logger, N, known, chunk_pairs=args.chunk_pairs, log_every=1000):
                for sl, probs in score_in_batches(logger, model, z, cu, cv,
                                                  batch_edges=args.batch_edges,
                                                  log_every_batches=args.log_every_batches):
                    uu = cu[sl].numpy(); vv = cv[sl].numpy()
                    for p, i, j in zip(probs, uu, vv):
                        if len(top_heap) < args.topk:
                            heapq.heappush(top_heap, (p, int(i), int(j)))
                        else:
                            if p > top_heap[0][0]:
                                heapq.heapreplace(top_heap, (p, int(i), int(j)))
                    total_scored += sl.stop - sl.start
                    if total_scored % 1_000_000 == 0:
                        logger.info(f"[all] scored {fmtn(total_scored)} pairs… ({mem_str()})")
        out = os.path.join(ART, "topk_predictions.csv")
        write_topk_csv(logger, rev_id, top_heap, out)
        sc = [s for (s, _, _) in top_heap]
        stats = summarize_scores(logger, np.array(sc, dtype=np.float32))
        summary = {
            "mode": args.mode, "N": N, "known_edges": len(known),
            "candidates": "ALL-streamed", "topk": int(args.topk),
            "batch_edges": int(args.batch_edges), "device": DEVICE, "score_stats": stats,
        }
        json.dump(summary, open(os.path.join(ART, "infer_summary.json"), "w"), indent=2)
        logger.info(f"[OK] Summary → {os.path.join(ART,'infer_summary.json')}")

if __name__ == "__main__":
    main()
