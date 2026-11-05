"""Bridge the read-only current CLC bank and exact schema constructor."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'raw_connector'))
from features import build_features as _build
P=json.loads(Path('/mnt/data/yxy/program/RAXO/experiments/pair_conflict_material_projection_20261101/config/protocol.json').read_text())
c=P['cases']['CLCXray_test'];C=json.loads(Path(c['C_feature_manifest']['path']).read_text());N=json.loads(Path(c['N_feature_manifest']['path']).read_text());A=json.loads(Path(c['auxiliary_feature_manifest']['path']).read_text())
def build_features(V,Bnative,Bfresh,rows,image_sizes,proto,cats):
 return _build(V,Bnative,Bfresh,rows,image_sizes,proto,cats,native_support=C['material_support_count'],aux_support=A['actual_material_support_counts'],original_support=N['actual_material_support_counts'],case='CLCXray_test')
