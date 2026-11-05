"""Frozen runtime primitives; native scalar/float operation order, no AP."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import ast,contextlib,datetime,hashlib,io,json,math,sys
from pathlib import Path
import numpy as np
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime')
PROJECT=Path('/mnt/data/yxy/program/RAXO')
OLD=Path('/mnt/data/yxy-codex/home/outputs/RAXO/P16_P25_revision_20261024');D=OLD/'validation'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def jsonable(x):
 if isinstance(x,np.ndarray):return jsonable(x.tolist())
 if isinstance(x,np.generic):return jsonable(x.item())
 if isinstance(x,float) and not math.isfinite(x):return None
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 return x
def save(p,x):
 p=Path(p);assert p.resolve().is_relative_to(OUT.resolve());p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(jsonable(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def signature(records):
 h=hashlib.sha256()
 for r in records:
  h.update((json.dumps([int(r['image_id']),int(r['category_id']),*(float(v).hex() for v in r['bbox']),float(r['score']).hex()],separators=(',',':'))+'\n').encode())
 return dict(count=len(records),sha256=h.hexdigest())
def canonical(records):return [{k:r[k] for k in ('image_id','category_id','bbox','score')} for r in records]
def pin():
 allowed=sorted(os.sched_getaffinity(0));chosen=list(range(72,80))
 assert set(chosen)<=set(allowed);os.sched_setaffinity(0,chosen);return chosen
def torch_setup(gpu=False):
 import torch
 torch.set_num_threads(8)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:assert torch.get_num_interop_threads()==1
 torch.set_grad_enabled(False);torch.manual_seed(20261024)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 if gpu:torch.cuda.set_device(0)
 return torch
def extract(path,names,scope,kind=ast.FunctionDef):
 path=Path(path);tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,kind) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
 return {name:scope[name] for name in names}
def native_modules():
 if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
 from rmdx import cache_fusion,preargmax_fusion,material
 return cache_fusion,preargmax_fusion,material
_MIXTURE=None
def mixture(V,M,Tv,Tm,alpha):
 # Exact accepted body, extracted from the frozen P16 producer (no module import
 # with its process-level CUDA environment changes and no producer save calls).
 import torch
 global _MIXTURE
 if _MIXTURE is None:_MIXTURE=extract(D/'calibration_and_eval.py',{'mixture'},dict(torch=torch,math=math))['mixture']
 return _MIXTURE(V,M,Tv,Tm,alpha)
class Decision:
 def __init__(self,prototype_meta,category_meta):
  self.protos=prototype_meta;self.cats=category_meta
  self.native,self.fusion,self.material=native_modules()
  self.lookup=self.native.build_dist_index_lookup(prototype_meta,category_meta)
  self.class_by_id={int(c['category_id']):int(c['class_index']) for c in category_meta}
  assert len(self.class_by_id)==len(category_meta)
  for p in prototype_meta:
   if p['is_negative']:assert p['material_class_index'] is None
   else:
    c=category_meta[int(p['material_class_index'])]
    assert c['category_name']==p['category_name'] and c['category_id']==p['category_id']
 def minimal(self,rows,F,capture=False,need_indices=False):
  """Native per-row argmax/canonical mean/Python strict UNC; required fields only."""
  import torch
  records=[];kept_rows=[]
  if capture:
   N=len(rows);states=dict(winner=np.empty(N,np.int64),category=np.full(N,-1,np.int64),negative=np.empty(N,bool),
    before_unc=np.empty(N,bool),after_unc=np.zeros(N,bool),inter=np.full(N,np.nan),intra=np.full(N,np.nan))
  for i,row in enumerate(rows):
   vector=F[i];winner=int(torch.argmax(vector).item());proto=self.protos[winner];negative=bool(proto['is_negative'])
   if capture:states['winner'][i]=winner;states['negative'][i]=negative;states['before_unc'][i]=not negative
   if negative:continue
   idx=self.lookup[winner];pred=vector[int(proto['canonical_index'])]
   mean=vector[idx].mean() if idx.numel()>0 else torch.tensor(0.)
   inter=float((pred-mean).item());keep=inter>.15
   if capture:
    states['category'][i]=int(proto['category_id']);states['inter'][i]=inter
    states['intra'][i]=float((vector[winner]-pred).item());states['after_unc'][i]=keep
   if keep:
    records.append(dict(image_id=int(row['image_id']),category_id=int(proto['category_id']),bbox=row['bbox'],score=float(row['detector_score'])))
    if capture or need_indices:kept_rows.append(i)
  return records,kept_rows,states if capture else None
 def apply(self,rows,V,B,kind,value,Tv=None,Tm=None,capture=False):
  if kind in ('visual','post'):
   records,kept,states=self.minimal(rows,V,capture,need_indices=kind=='post')
   if kind=='post':
    assert B is not None
    records=self.rescore(records,kept,B,value)
   return records,states
  assert B is not None
  if kind=='visual_wrapper_as_is':
   cache=dict(visual_similarity=V,material_class_similarity=B,prototype_meta=self.protos,category_meta=self.cats,proposal_meta=rows)
   # Preserve the actual wrapper's otherwise-unneeded allocations/diagnostics.
   F,M,gate=self.native.fuse_from_cache(cache,mode='visual_only',lambda_material=0.,lambda_reliability=0.)
   out=self.native.detections_from_fusion(cache,F,M,gate,mode='visual_only',lambda_material=0.,lambda_reliability=0.)
   records=canonical([r for r in out if float(r['dist_cosine_inter'])>.15])
   states=self.minimal(rows,F,True)[2] if capture else None
   return records,states
  # This full-J expansion is always inside the additive/mixture decision call.
  M=self.native.material_class_to_prototype(B,self.protos)
  if kind=='additive':F=self.fusion.fuse_similarity_material(V,M,value)
  elif kind=='mixture':F=mixture(V,M,Tv,Tm,value)
  else:raise ValueError(kind)
  return self.minimal(rows,F,capture)[::2]  # records, states
 def post_work(self,records,kept,B):
  return [dict(r,rmdx_material_score=(1.+float(B[int(i),self.class_by_id[int(r['category_id'])]].item()))/2.) for r,i in zip(records,kept)]
 def rescore(self,records,kept,B,beta):
  work=self.post_work(records,kept,B)
  return canonical(self.material.apply_material_rescore(work,beta,replace_score=True))
def load_cached(family,kind):
 import torch
 f=family
 if f['format']=='npy':V=torch.from_numpy(np.array(np.load(f['V'],mmap_mode='r'),copy=True))
 else:V=torch.load(f['V'],map_location='cpu').float()
 B=None
 if kind!='visual':
  if f['format']=='npy':B=torch.from_numpy(np.array(np.load(f['B'],mmap_mode='r'),copy=True))
  else:B=torch.load(f['B'],map_location='cpu').float()
 rows=[json.loads(line) for line in Path(f['rows']).read_text().splitlines() if line.strip()]
 p=read(f['prototype_meta']);c=read(f['category_meta']);assert V.shape==(f['N'],f['J']) and V.dtype==torch.float32
 if B is not None:assert B.shape==(f['N'],f['C']) and B.dtype==torch.float32
 return rows,V,B,Decision(p,c)
def imports_receipt():
 rows=[]
 for name,module in sorted(sys.modules.items()):
  file=getattr(module,'__file__',None)
  if file and name.startswith(('torch','torchvision','groundingdino','sam2','dinov2','transformers','pycocotools','rmdx','cv2')):
   p=Path(file)
   if p.is_file():rows.append(dict(module=name,**ref(p.resolve())))
 mapped=[]
 for line in Path('/proc/self/maps').read_text().splitlines():
  fields=line.split()
  if len(fields)>=6 and fields[-1].startswith('/') and '.so' in fields[-1]:
   p=Path(fields[-1])
   if p.is_file():mapped.append(p.resolve())
 return dict(Python=sys.executable,modules=rows,actual_mapped_shared_libraries=[ref(p) for p in sorted(set(mapped))])
