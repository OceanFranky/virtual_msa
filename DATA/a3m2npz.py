from pathlib import Path
from boltz.data.parse.a3m import parse_a3m

a3m_path = Path("/data/wenhao/Protein-Hunter-server/MSA/9OZL_A_clean_fixed.a3m")
out_path = Path("/data/wenhao/Protein-Hunter-server/MSA/9OZL.npz")

msa = parse_a3m(a3m_path, taxonomy=None, max_seqs=None)
msa.dump(out_path)

print("saved:", out_path)
