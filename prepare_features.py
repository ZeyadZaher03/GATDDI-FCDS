import json, os, numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

DATA_DIR = os.getenv("DATA_DIR", "data")
ARTIFACTS = os.getenv("ARTIFACTS", "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

def morgan_fp(smiles: str, radius=2, nbits=1024):
    mol = Chem.MolFromSmiles(smiles)
    arr = np.zeros((nbits,), dtype=np.float32)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

drugs = pd.read_csv(f"{DATA_DIR}/drugs.csv")  # columns: drug_id,name,smiles
drugs = drugs.drop_duplicates(subset=["drug_id"]).reset_index(drop=True)
id_map = {did: i for i, did in enumerate(drugs["drug_id"].tolist())}
X = np.vstack([morgan_fp(s) for s in drugs["smiles"].astype(str).tolist()])

np.save(f"{ARTIFACTS}/x.npy", X)
with open(f"{ARTIFACTS}/id_map.json", "w") as f:
    json.dump(id_map, f)
print(f"Saved features: {X.shape}, nodes={len(id_map)}")
