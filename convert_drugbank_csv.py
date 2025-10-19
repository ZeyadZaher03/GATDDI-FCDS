import os, re, json, ast, pandas as pd
from typing import List, Optional, Iterable

IN_PATH  = os.getenv("IN_CSV",  "merged_output.csv")  # your file
OUT_DIR  = os.getenv("OUT_DIR", "data")
os.makedirs(OUT_DIR, exist_ok=True)

ID_RE = re.compile(r"(DB\d+)", re.IGNORECASE)

def normalize_id(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    m = ID_RE.search(s.strip())
    if not m:
        return None
    did = m.group(1).upper()
    # Normalize to DBxxxxx (5+ digits acceptable)
    return did

def parse_list_cell(cell) -> List[str]:
    """
    Robustly parse 'drug-interactions' variations:
    - JSON: '["DB0001","DB0002"]'
    - Python repr: "['DB0001','DB0002']"
    - Delimited: 'DB0001; DB0002' or 'DB0001,DB0002' or 'DB0001|DB0002'
    - Empty/NaN -> []
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        raw = cell
    else:
        s = str(cell).strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return []
        # Try JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                raw = obj
            else:
                raw = [obj]
        except Exception:
            # Try Python literal (e.g., "['DB0001', 'DB0002']")
            try:
                obj = ast.literal_eval(s)
                raw = obj if isinstance(obj, (list, tuple, set)) else [obj]
            except Exception:
                # Fallback: split by common delimiters
                for delim in [";", "|", ","]:
                    if delim in s:
                        raw = [t.strip() for t in s.split(delim)]
                        break
                else:
                    raw = [s]
    # Normalize & filter
    out = []
    for r in raw:
        rid = normalize_id(str(r))
        if rid:
            out.append(rid)
    # Deduplicate while preserving order
    seen = set(); uniq = []
    for rid in out:
        if rid not in seen:
            seen.add(rid); uniq.append(rid)
    return uniq

def main():
    df = pd.read_csv(IN_PATH)
    # Column names we care about (your CSV shows them exactly like this)
    # 'drugbank_id', 'name', 'smiles', 'drug-interactions'
    assert "drugbank_id" in df.columns, "Missing 'drugbank_id' column"
    assert "name"        in df.columns, "Missing 'name' column"
    assert "smiles"      in df.columns, "Missing 'smiles' column"
    assert "drug-interactions" in df.columns, "Missing 'drug-interactions' column"

    # Normalize IDs & minimal drug table
    df["drug_id"] = df["drugbank_id"].astype(str).apply(normalize_id)
    df = df[~df["drug_id"].isna()]
    drugs = df[["drug_id", "name", "smiles"]].drop_duplicates("drug_id").reset_index(drop=True)
    drugs.to_csv(os.path.join(OUT_DIR, "drugs.csv"), index=False)

    # Build edges from drug-interactions
    # Each row lists neighbors for that drug_id
    pairs = []
    id_set = set(drugs["drug_id"].tolist())

    for _, row in df.iterrows():
        src = row["drug_id"]
        neighs = parse_list_cell(row.get("drug-interactions", None))
        for dst in neighs:
            if dst == src:
                continue
            if dst not in id_set:
                # skip edges to unknown drugs (not present in this dump)
                continue
            # undirected canonicalization
            a, b = (src, dst) if src < dst else (dst, src)
            pairs.append((a, b))

    # Deduplicate edges
    if pairs:
        edges = pd.DataFrame(pairs, columns=["head_id", "tail_id"]).drop_duplicates()
        edges["label"] = 1
    else:
        edges = pd.DataFrame(columns=["head_id","tail_id","label"])

    edges.to_csv(os.path.join(OUT_DIR, "ddi.csv"), index=False)

    # Quick stats
    print(f"[OK] Wrote {len(drugs)} drugs to data/drugs.csv")
    print(f"[OK] Wrote {len(edges)} edges to data/ddi.csv")

if __name__ == "__main__":
    main()
