#!/usr/bin/env python3
import argparse,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from robust_model import PROTOCOL_SHA,predict_checkpoint,read_protocol,read,ref,sha,verify,write,now
def main():
 p=argparse.ArgumentParser()
 for n in ['input','feature-manifest','checkpoint','protocol','output-dir']:p.add_argument('--'+n,required=True)
 p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8192);p.add_argument('--input-conf-factor',type=float,default=1.);p.add_argument('--entropy-shift',type=float,default=0.);p.add_argument('--profile',default='natural');a=p.parse_args();pp=Path(a.protocol).resolve();P=read_protocol(pp);src=Path(a.input).resolve();fmp=Path(a.feature_manifest).resolve();cp=Path(a.checkpoint).resolve();o=Path(a.output_dir).resolve();fm=read(fmp)
 if o.exists()and any(o.iterdir()):raise FileExistsError('New B inference output required')
 if fm['outputs'][src.name]!=ref(src):raise ValueError('Frozen source features differ')
 case=fm.get('case',fm.get('dataset'));declared=P['cases'][case]
 if declared['feature_manifest']!=ref(fmp)or declared['features']!=ref(src):raise ValueError('Frozen protocol case source differs')
 profiles=[x for x in P['stress_profiles']if x['id']==a.profile]
 if len(profiles)!=1 or profiles[0]['input_conf_factor']!=a.input_conf_factor or profiles[0]['entropy_shift']!=a.entropy_shift:raise ValueError('Q-input stress outside frozen profiles')
 if case!='CLC_tune'and(a.input_conf_factor!=1. or a.entropy_shift!=0.):raise ValueError('Stress is calibration-only')
 names=read(verify(fm['outputs']['feature_names.json']))
 if names!=P['feature_schema']['names']:raise ValueError('Frozen feature schema differs')
 x=np.load(src,mmap_mode='r',allow_pickle=False)
 if x.dtype!=np.float32 or list(x.shape)!=fm['shape']or x.shape[1:]!=(5,15)or not fm['candidate_row_key_sha256']:raise ValueError('Candidate source shape/rows differ')
 if not np.isfinite(a.input_conf_factor)or not np.isfinite(a.entropy_shift)or a.input_conf_factor<0 or a.batch_size<1:raise ValueError('Malformed q-only stress arguments')
 o.mkdir(parents=True,exist_ok=True);start=time.perf_counter();q,conf=predict_checkpoint(cp,x,names,a.device,a.batch_size,P,a.input_conf_factor,a.entropy_shift);torch.cuda.synchronize()
 if q.dtype!=np.float32 or q.shape!=(len(x),5)or not np.isfinite(q).all()or np.any((q<0)|(q>1)):raise ValueError('Malformed B q')
 with(o/'quality_scores.npz').open('xb')as f:np.savez_compressed(f,quality=q)
 result=dict(status='COMPLETE_LABEL_FREE_B_INFERENCE',completed_utc=now(),PID=os.getpid(),argv=sys.argv,case=fm.get('case',fm.get('dataset')),profile=a.profile,input_conf_factor=a.input_conf_factor,entropy_shift=a.entropy_shift,q_input_stress_only=True,original_raw_output_scores_changed=False,recipe_id=conf['recipe_id'],variant=conf['variant'],seed=conf['seed'],protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,checkpoint=ref(cp),input=ref(src),feature_manifest=ref(fmp),candidate_row_key_sha256=fm['candidate_row_key_sha256'],shape=list(q.shape),dtype='float32',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_name=torch.cuda.get_device_name(),device=a.device,elapsed_seconds=time.perf_counter()-start,output=ref(o/'quality_scores.npz'),output_sha256=sha(o/'quality_scores.npz'),generator=ref(__file__),model_module=ref(Path(__file__).resolve().parents[1]/'src/robust_model.py'),GT_opened=False,labels_read=False,AP_computed=False,recipe_selected=False)
 write(o/'inference.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
