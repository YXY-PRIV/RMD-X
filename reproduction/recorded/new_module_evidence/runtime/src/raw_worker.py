#!/usr/bin/env python3
"""Persistent real CLC main/SAM workers; JSON IPC; no GT labels or AP."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0')
import argparse,collections,contextlib,copy,gc,io,json,random,subprocess,sys,time,types
from pathlib import Path
from multiprocessing.connection import Client,Listener
sys.path.insert(0,str(Path(__file__).parent))
from raw_common import *
AUTH=b'p28-local-runtime-fixed'
GD=Path('/mnt/data/yxy/external/GroundingDINO_clean');SAM=Path('/mnt/data/yxy/external/SAM2_clean')
HUB=Path('/data/raxo/models/torch_cache_for_rerun/hub/facebookresearch_dinov2_main')
SAM_PY='/mnt/data/yxy/envs/xowtta_sam2/bin/python'
def send(conn,obj):conn.send_bytes(json.dumps(jsonable(obj),separators=(',',':'),allow_nan=False).encode())
def receive(conn):return json.loads(conn.recv_bytes())
def gpu_memory(torch):
 torch.cuda.synchronize(0)
 return {k:int(f(0)) for k,f in [('allocated',torch.cuda.memory_allocated),('reserved',torch.cuda.memory_reserved),('max_allocated',torch.cuda.max_memory_allocated),('max_reserved',torch.cuda.max_memory_reserved)]}
class Clock:
 def __init__(self,torch,track):self.torch=torch;self.track=track;self.stages=collections.defaultdict(int)
 def call(self,name,fun):
  self.torch.cuda.synchronize(0)
  t=time.perf_counter_ns() if self.track else 0
  value=fun();self.torch.cuda.synchronize(0)
  if self.track:self.stages[name]+=time.perf_counter_ns()-t
  return value
def sam_worker(conn,trial,track):
 torch=torch_setup(True);sys.path.insert(0,str(SAM));sys.path.insert(0,str(OLD/'deps/sam_pycocotools'))
 from sam2.build_sam import build_sam2
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 import pycocotools.mask as mu
 from PIL import Image
 p=read(D/'protocols/upstream_protocol.json');clock=Clock(torch,track)
 def load():
  model=build_sam2(p['SAM2']['config'],None,device='cuda',mode='eval').float()
  checkpoint=torch.load(p['SAM2']['checkpoint'],map_location='cpu',weights_only=True)['model']
  loaded=model.load_state_dict(checkpoint,strict=True);assert not loaded.missing_keys and not loaded.unexpected_keys
  return model,SAM2ImagePredictor(model),dict(strict=True,model_keys=len(model.state_dict()),loaded_keys=len(checkpoint))
 model,predictor,loaded=clock.call('SAM_model_load',load)
 send(conn,dict(ready=True,pid=os.getpid(),Python=sys.executable,torch_version=torch.__version__,load=loaded,
  startup_stages_ns=dict(clock.stages),GPU=gpu_memory(torch),parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()})))
 torch.cuda.reset_peak_memory_stats(0)
 while True:
  request=receive(conn);op=request['op']
  if op=='close':send(conn,dict(closed=True,GPU=gpu_memory(torch)));break
  if op=='imports':send(conn,imports_receipt());continue
  if op=='reset':
   previous=gpu_memory(torch);torch.cuda.reset_peak_memory_stats(0);send(conn,dict(previous=previous,current=gpu_memory(torch)));continue
  if op=='memory':send(conn,gpu_memory(torch));continue
  assert op=='masks'
  clock=Clock(torch,track);rows=request['rows']
  rgb=clock.call('SAM_decode',lambda:np.asarray(Image.open(request['image_path']).convert('RGB')))
  result=[]
  with torch.inference_mode():
   clock.call('SAM_set_image',lambda:predictor.set_image(rgb))
   for lo in range(0,len(rows),32):
    chunk=rows[lo:lo+32]
    boxes=np.asarray([[r['bbox'][0],r['bbox'][1],r['bbox'][0]+r['bbox'][2],r['bbox'][1]+r['bbox'][3]] for r in chunk],dtype=np.float32)
    masks,quality,_=clock.call('SAM_box_masks',lambda:predictor.predict(box=boxes,multimask_output=False))
    if masks.ndim==3:masks=masks[:,None]
    assert masks.shape==(len(chunk),1,rgb.shape[0],rgb.shape[1])
    def encode():
     encoded=[]
     for row,mask,q in zip(chunk,masks[:,0],np.asarray(quality).reshape(-1)):
      a=np.asarray(mask,dtype=np.uint8);rle=mu.encode(np.asfortranarray(a));rle['counts']=rle['counts'].decode('ascii')
      encoded.append(dict(id=row['annotation_id'],image_id=row['image_id'],bbox=row['bbox'],score=row['detector_score'],mask=rle,SAM_predicted_iou=float(q),SAM_empty=not bool(a.any())))
     return encoded
    result.extend(clock.call('SAM_RLE_encode',encode))
  send(conn,dict(annotations=result,stages_ns=dict(clock.stages),GPU=gpu_memory(torch)))
def main_worker(conn,trial,boundary,method,track):
 torch=torch_setup(True)
 import torch.nn.functional as F
 import torchvision.transforms as T
 import cv2;cv2.setNumThreads(1)
 import pycocotools.mask as mask_util
 from pycocotools.coco import COCO
 from PIL import Image
 from torch.utils.data import Dataset
 p=read(D/'protocols/upstream_protocol.json');clock=Clock(torch,track);sam=None;sam_process=None;sam_load=None
 cats=read(PROJECT/'outputs_rmdx/cache_multidataset/CLCXray_groundingDINO_100_0_full/category_meta.json');protos=read(PROJECT/'outputs_rmdx/cache_multidataset/CLCXray_groundingDINO_100_0_full/prototype_meta.json');decision=Decision(protos,cats)
 transform=T.Compose([T.Resize((224,224),interpolation=T.InterpolationMode.BICUBIC),T.ToTensor(),T.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
 DatasetClass=extract(PROJECT/'raxo/dataset.py',{'CocoPrototypes_with_masks'},dict(Dataset=Dataset,random=random,os=os,Image=Image,np=np,torch=torch,mask_util=mask_util,T=T,cv2=cv2,F=F),__import__('ast').ClassDef)['CocoPrototypes_with_masks']
 def load_dino():
  model=torch.hub.load(str(HUB),'dinov2_vitb14_reg',source='local',pretrained=False)
  checkpoint=torch.load(p['DINO']['checkpoint'],map_location='cpu',weights_only=True)
  status=model.load_state_dict(checkpoint,strict=True);assert not status.missing_keys and not status.unexpected_keys
  return model.eval().float().cuda(),dict(strict=True,model_keys=len(model.state_dict()),loaded_keys=len(checkpoint))
 model,dino_load=clock.call('DINO_model_load',load_dino)
 def load_bank():
  source=torch.load(D/'banks/visual_prototypes.pt',map_location='cpu',weights_only=True)
  proxy=types.SimpleNamespace(load=lambda path:source,stack=torch.stack)
  assembler=extract(PROJECT/'rmdx/rmdx_main_with_masks_prop.py',{'assemble_prototypes'},dict(os=os,torch=proxy))['assemble_prototypes']
  P,names,keys=assembler(str(D/'banks/visual_prototypes.pt'),[c['category_name'] for c in cats])
  assert keys==[q['prototype_key'] for q in protos] and P.shape==(733,768)
  return P.cuda()
 P=clock.call('visual_bank_load',load_bank)
 appearance=None
 if method!='visual':
  def load_appearance():
   bank=read(PROJECT/'outputs_rmdx/material_multidataset/banks/CLCXray_material_bank_transfer_from_det_compass_support.json');names=[c['category_name']for c in cats];lookup=decision.fusion.build_category_material_matrix(names,names,bank);stats=lookup['stats']
   vectors=[lookup['category_features'][name]for name in names]
   return dict(stats=stats,means=vectors,z=[decision.material.zscore_feature(q,stats) for q in vectors])
  appearance=clock.call('appearance_bank_load',load_appearance)
 gdmodel=None;gdtransform=None
 if boundary=='C':
  sys.path.insert(0,str(GD))
  from groundingdino.models import build_model
  from groundingdino.util.slconfig import SLConfig
  from groundingdino.util.misc import clean_state_dict
  import groundingdino.datasets.transforms as GT
  def load_gd():
   args=SLConfig.fromfile(p['detector']['config']);args.device='cuda';m=build_model(args)
   s=clean_state_dict(torch.load(p['detector']['checkpoint'],map_location='cpu',weights_only=False)['model']);status=m.load_state_dict(s,strict=False)
   assert not status.missing_keys
   return m.eval().float().cuda(),dict(missing=list(status.missing_keys),unexpected=list(status.unexpected_keys),model_keys=len(m.state_dict()),loaded_keys=len(s)-len(status.unexpected_keys))
  gdmodel,gd_load=clock.call('GDINO_model_load',load_gd)
  gdtransform=GT.Compose([GT.RandomResize([800],max_size=1333),GT.ToTensor(),GT.Normalize([.485,.456,.406],[.229,.224,.225])])
  (OUT/'ipc').mkdir(exist_ok=True);socket=OUT/'ipc'/f'sam_{os.getpid()}.sock';assert len(str(socket).encode())<108
  listener=Listener(str(socket),family='AF_UNIX',authkey=AUTH)
  command=[SAM_PY,'-B',str(Path(__file__)),'--role','sam','--socket',str(socket),'--trial',str(trial),'--boundary',boundary,'--method',method]+(['--track'] if track else [])
  sam_process=subprocess.Popen(command,stdout=(trial/'SAM_worker.log').open('w'),stderr=subprocess.STDOUT)
  sam=listener.accept();sam_load=receive(sam);listener.close();socket.unlink(missing_ok=True);assert sam_load['ready']
 else:gd_load=None
 temperatures=None
 from raw_deployment import Deployment
 from raw_features import build_features
 quality_model=clock.call('quality_ensemble_load',lambda:Deployment('cuda:0',method))

 send(conn,dict(ready=True,pid=os.getpid(),sam_pid=None if sam_process is None else sam_process.pid,Python=sys.executable,torch_version=torch.__version__,
  startup_stages_ns=dict(clock.stages),GPU=gpu_memory(torch),DINO_load=dino_load,GDINO_load=gd_load,SAM_load=sam_load,
  appearance_bank_loaded=appearance is not None,prototype_shape=list(P.shape),quality_parameters=quality_model.parameter_count,quality_unique_resident_parameters=quality_model.unique_resident_parameter_count,quality_checkpoint_refs=quality_model.refs,parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()})))
 torch.cuda.reset_peak_memory_stats(0);buffer=[];last_capture=None;last_meta=None;metadata_history=[]
 def detect(entry,clock):
  rgb=clock.call('GDINO_decode',lambda:Image.open(entry['image']['path']).convert('RGB'))
  tensor=clock.call('GDINO_transform',lambda:gdtransform(rgb,None)[0])
  x=clock.call('GDINO_H2D',lambda:tensor[None].cuda())
  with torch.inference_mode():output=clock.call('GDINO_forward',lambda:gdmodel(x,captions=[p['detector']['caption']]))
  logits,boxes=clock.call('GDINO_D2H',lambda:(output['pred_logits'][0].sigmoid().cpu(),output['pred_boxes'][0].cpu()))
  def postprocess():
   assert bool(torch.isfinite(logits).all()) and bool(torch.isfinite(boxes).all())
   scores=logits.max(1).values;indices=torch.nonzero(scores>.10).reshape(-1).tolist();candidates=[]
   for query in indices:
    box=boxes[query]*torch.tensor([rgb.width,rgb.height,rgb.width,rgb.height],dtype=torch.float32)
    x0=(box[0]-box[2]/2).clamp(0,rgb.width);y0=(box[1]-box[3]/2).clamp(0,rgb.height)
    x1=(box[0]+box[2]/2).clamp(0,rgb.width);y1=(box[1]+box[3]/2).clamp(0,rgb.height)
    candidates.append(dict(query_index=query,detector_score=float(scores[query]),bbox=[float(x0),float(y0),float(x1-x0),float(y1-y0)]))
   retained=[]
   for row in sorted(candidates,key=lambda r:(-r['detector_score'],r['query_index'])):
    a=row['bbox'];reject=False
    for old in retained:
     b=old['bbox'];intersection=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))*max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));union=a[2]*a[3]+b[2]*b[3]-intersection
     if union>0 and intersection/union>.8:reject=True;break
    if not reject:retained.append(row)
   result=[]
   for row in retained[:100]:
    if all(v>=2 for v in row['bbox']):result.append(dict(row,annotation_id=len(result)+1,image_id=entry['original_image_id'],original_image_id=entry['original_image_id'],file_name=entry['file_name']))
   return result
  return clock.call('GDINO_NMS_cap_native_filter',postprocess)
 def features(entry,rows,annotations,clock,capture):
  def dataset():
   coco=COCO();coco.dataset=dict(images=[dict(id=entry['original_image_id'],file_name=entry['file_name'])],annotations=[dict(a,category_id=-1) for a in annotations],categories=[dict(id=-1,name='UNASSIGNED_PROPOSAL_LOADER_ONLY')]);coco.createIndex()
   ds=DatasetClass(coco,str(Path(entry['image']['path']).parent),transform);assert len(ds)==len(rows);return ds
  with contextlib.redirect_stdout(io.StringIO()):ds=clock.call('native_dataset_metadata',dataset)
  visual=[];embeddings=[]
  for lo in range(0,len(rows),64):
   def crop():
    batch=[ds[i] for i in range(lo,min(lo+64,len(rows)))];return torch.stack([x[0] for x in batch]),torch.stack([x[2] for x in batch])
   images,masks=clock.call('native_per_proposal_decode_crop_RLE_resize_mask',crop)
   images,weights=clock.call('DINO_H2D',lambda:(images.cuda(),masks.cuda().unsqueeze(-1)))
   with torch.inference_mode():
    tokens=clock.call('DINO_forward',lambda:model.get_intermediate_layers(images)[0])
    q=clock.call('masked_patch_pooling',lambda:((tokens*weights).sum(dim=1,keepdim=True)/weights.sum(dim=1,keepdim=True).clamp(min=1)).squeeze(1))
    v=clock.call('visual_cosine_fullJ',lambda:F.cosine_similarity(q.unsqueeze(1),P.unsqueeze(0),dim=-1))
    visual.append(clock.call('visual_D2H',lambda:v.cpu()))
    if capture:embeddings.append(q.cpu())
  V=torch.cat(visual) if visual else torch.empty((0,733),dtype=torch.float32)
  return V,(torch.cat(embeddings).numpy() if embeddings else np.empty((0,768),np.float32)) if capture else None
 def material(entry,rows,clock,capture):
  rgb=clock.call('appearance_decode',lambda:np.asarray(Image.open(entry['image']['path']).convert('RGB'),dtype=np.uint8))
  q=clock.call('exact22_full_bbox',lambda:[decision.material.extract_colour_feature_from_array(rgb,r['bbox'],mode='bbox_only') for r in rows])
  def cosine():
   out=[]
   for feature in q:
    z=decision.material.zscore_feature(feature,appearance['stats']);out.append([decision.material.cosine_similarity(z,p) for p in appearance['z']])
   return torch.tensor(out,dtype=torch.float32).reshape(len(rows),12)
  B=clock.call('query_zscore_scalar_cosine',cosine)
  return B,np.asarray(q,dtype=np.float64).reshape(len(rows),22) if capture else None
 while True:
  request=receive(conn);op=request['op']
  if op=='close':
   sm=None
   if sam is not None:send(sam,dict(op='close'));sm=receive(sam);sam.close();sam_process.wait();assert sam_process.returncode==0
   send(conn,dict(closed=True,GPU=gpu_memory(torch),SAM=sm));break
  if op=='imports':
   si=None
   if sam is not None:send(sam,dict(op='imports'));si=receive(sam)
   send(conn,dict(main=imports_receipt(),SAM=si));continue
  if op=='reset':
   prior=gpu_memory(torch);torch.cuda.reset_peak_memory_stats(0);sp=None
   if sam is not None:send(sam,dict(op='reset'));sp=receive(sam)
   send(conn,dict(main_previous=prior,main_current=gpu_memory(torch),SAM=sp));continue
  if op=='memory':
   sp=None
   if sam is not None:send(sam,dict(op='memory'));sp=receive(sam)
   send(conn,dict(main=gpu_memory(torch),SAM=sp));continue
  if op=='capture':
   assert last_capture is not None;send(conn,last_capture);last_capture=None;continue
  if op=='metadata':
   assert last_meta is not None;send(conn,last_meta);last_meta=None;continue
  if op=='drain_metadata':
   send(conn,metadata_history);metadata_history=[];last_meta=None;continue
  if op=='clear_metadata':
   metadata_history=[];last_meta=None;send(conn,dict(cleared=True));continue
  if op=='flush_post':
   assert method=='post';work=[r for batch in buffer for r in batch];clock=Clock(torch,track)
   final=clock.call('whole_cohort_post_mean_and_scores',lambda:canonical(decision.material.apply_material_rescore(work,1.,replace_score=True)))
   mean=sum(r['rmdx_material_score'] for r in work)/len(work) if work else 0.;buffer=[]
   send(conn,dict(records=final,cohort_mean_u=mean,stages_ns=dict(clock.stages)));continue
  assert op=='image';entry=request['entry'];capture=request.get('capture',False);clock=Clock(torch,track)
  assert not(track and capture),'Diagnostic feature/state capture is outside all valid timed trials'
  if boundary=='C':
   rows=detect(entry,clock)
   def sam_call():send(sam,dict(op='masks',image_path=entry['image']['path'],rows=rows));return receive(sam)
   masked=clock.call('SAM_roundtrip_IPC_wait',sam_call);annotations=masked['annotations']
   for k,v in masked['stages_ns'].items():clock.stages['SAM_diagnostic_'+k]=v
  else:
   fixed=clock.call('fixed_proposal_mask_read',lambda:read(entry['fixed_masks']['path']));rows=fixed['rows'];annotations=fixed['annotations']
  V,emb=features(entry,rows,annotations,clock,capture)
  B=None;q22=None
  if appearance is not None:B,q22=material(entry,rows,clock,capture)
  def quality_finalize():
   data=build_features(V.numpy(),B.numpy(),B.numpy(),rows,{entry['original_image_id']:(entry['width'],entry['height'])},protos,cats)
   result=quality_model.run(data,method,capture=capture) if len(data['M']) else dict(records=[],stages_ns={},q=np.empty((0,2),np.float32),q0=np.empty((0,2),np.float32),selected=np.empty(0,np.int8),post=np.empty(0,np.float64))
   return result,data
  result,data=clock.call('complete_P5_quality_ensemble_and_Gaussian',quality_finalize)
  response_records=result['records'];records=response_records;states=None
  if capture:
   assert bool(torch.isfinite(V).all()) and (B is None or bool(torch.isfinite(B).all())) and np.isfinite(emb).all()
   last_capture=dict(V=V.numpy().tolist(),B=None if B is None else B.numpy().tolist(),embedding=emb.tolist(),exact22=None if q22 is None else q22.tolist(),states=states,rows=rows,visual_records=records,q=result['q'].tolist(),q0=result['q0'].tolist(),selected=result['selected'].tolist(),post=result['post'].tolist())
  last_meta=dict(proposal_rows=rows,mask_annotation_ids=[a['id'] for a in annotations])
  metadata_history.append(dict(original_image_id=entry['original_image_id'],**last_meta))
  send(conn,dict(records=response_records,proposal_count=len(rows),stages_ns=dict(clock.stages),buffered=False))
  del V,B;gc.collect()
def main():
 p=argparse.ArgumentParser();p.add_argument('--role',choices=['main','sam'],required=True);p.add_argument('--socket',required=True);p.add_argument('--trial',required=True);p.add_argument('--boundary',choices=['B','C'],required=True);p.add_argument('--method',choices=['strong','pair'],required=True);p.add_argument('--track',action='store_true');a=p.parse_args();pin()
 trial=Path(a.trial);conn=Client(a.socket,family='AF_UNIX',authkey=AUTH)
 if a.role=='sam':sam_worker(conn,trial,a.track)
 else:main_worker(conn,trial,a.boundary,a.method,a.track)
 conn.close()
if __name__=='__main__':main()
