#!/usr/bin/env python3
import argparse,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.union_model import infer
from src.union_data import PSHA,read,ref,sha,verify,atomic,now
def main():
    p=argparse.ArgumentParser()
    for name in ['feature-manifest','checkpoint','protocol','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8192);a=p.parse_args();pp=Path(a.protocol).resolve()
    if PSHA=='PENDING'or sha(pp)!=PSHA:raise ValueError('Frozen union protocol required')
    fmp=Path(a.feature_manifest).resolve();F=read(fmp);X=np.load(verify(F['outputs']['candidate_features.npy']),allow_pickle=False,mmap_mode='r');M=np.load(verify(F['outputs']['candidate_valid.npy']),allow_pickle=False);names=read(verify(F['outputs']['feature_names.json']));out=Path(a.output_dir).resolve()
    if out.exists()and any(out.iterdir()):raise FileExistsError('New inference directory required')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();q,c=infer(a.checkpoint,X,M,names,a.device,a.batch_size);torch.cuda.synchronize()
    if q.dtype!=np.float32 or q.shape!=M.shape or np.any(q[~M]!=0)or not np.isfinite(q).all()or np.any((q<0)|(q>1)):raise ValueError('Malformed union q')
    with(out/'quality_scores.npz').open('xb')as f:np.savez_compressed(f,quality=q)
    r=dict(status='COMPLETE_LABEL_FREE_SHARED_UNION_INFERENCE',completed_utc=now(),PID=os.getpid(),argv=sys.argv,case=F['case'],variant=c['variant'],seed=c['seed'],protocol=ref(pp),protocol_sha256=PSHA,feature_manifest=ref(fmp),candidate_row_key_sha256=F['candidate_row_key_sha256'],checkpoint=ref(a.checkpoint),output=ref(out/'quality_scores.npz'),output_sha256=sha(out/'quality_scores.npz'),shape=list(q.shape),dtype='float32',invalid_quality_exact0=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=a.device,GPU_name=torch.cuda.get_device_name(),elapsed_seconds=time.perf_counter()-start,GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False);atomic(out/'inference.json',r);print(json.dumps(r),flush=True)
if __name__=='__main__':main()
