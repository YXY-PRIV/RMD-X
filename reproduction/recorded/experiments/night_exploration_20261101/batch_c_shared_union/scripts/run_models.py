#!/usr/bin/env python3
"""Dedicated GPU1 worker:9 shared union models,63 inferences,21 means."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.union_data import CODE,OUT,PSHA,read,sha,ref,verify,atomic,now,frozen
from src.union_model import GPU_UUID,VARIANTS
def wait_complete(path):
    while True:
        if path.exists():
            r=read(path)
            if r.get('status')=='FAILED':raise RuntimeError(str(path)+' failed')
            if r.get('status')=='COMPLETE':return r
        time.sleep(2)
def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',default=str(CODE/'config/protocol.json'));a=p.parse_args();pp=Path(a.protocol).resolve();P,pr,_=frozen(pp)
    for d in ['automation','models','training','inference','ensembles','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
    lock=(OUT/'automation/model_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    paths=['src/union_data.py','src/union_model.py','scripts/train_union.py','scripts/infer_union.py','scripts/run_models.py'];sources={n:ref(CODE/n)for n in paths};freeze=OUT/'automation/model_code_freeze.json'
    if freeze.exists():raise FileExistsError('Preserve any prior C worker attempt before explicit repair')
    atomic(freeze,dict(status='FROZEN_BEFORE_FIRST_MODEL_RUN',protocol=pr,created_utc=now(),sources=sources))
    state=dict(status='WAITING_UNION_TRAINING_DATA',PID=os.getpid(),started_utc=now(),protocol=pr,active=None,completed_trainings=0,completed_inferences=0,completed_ensembles=0)
    def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_worker_status.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU_UUID,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
    def run(name,argv):
        if sources!={n:ref(CODE/n)for n in paths}:raise ValueError('Frozen C model source changed')
        log=OUT/'logs'/(name+'.log')
        with log.open('x')as stream:
            proc=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=Path('/mnt/data/yxy/program/RAXO'));launch=dict(PID=proc.pid,worker_PID=os.getpid(),argv=argv,started_utc=now(),log=str(log),CUDA_VISIBLE_DEVICES=GPU_UUID,protocol=pr);atomic(OUT/'automation'/('launch_'+name+'.json'),launch);update(active=launch);rc=proc.wait()
        update(active=None)
        if rc:raise RuntimeError(f'{name} exit{rc}: {log}')
        return dict(launch,exit_code=0,completed_utc=now())
    try:
        update();wait_complete(OUT/'training/data_status.json');inputpath=OUT/'training/union_train_val.npz';ts=dict(status='RUNNING',protocol=pr,input=ref(inputpath),runs=[],new_GT_annotations_read=False,threshold_selected=False)
        for variant in VARIANTS:
            for seed in [17,29,43]:
                update(status='RUNNING_ACTUAL_CUDA_TRAINING');d=OUT/'models'/variant/f'seed{seed}';job=run(f'train_{variant}_{seed}',[sys.executable,'-B',str(CODE/'scripts/train_union.py'),'--input',str(inputpath),'--protocol',str(pp),'--variant',variant,'--seed',str(seed),'--output-dir',str(d),'--device','cuda:0']);m=read(d/'metrics.json');c=read(d/'run_config.json')['config']
                if m['status']!='COMPLETE'or not m['actual_CUDA']or m['environment']['CUDA_VISIBLE_DEVICES']!=GPU_UUID or c['input']!=ref(inputpath)or c['protocol']!=pr or(c['variant'],c['seed'])!=(variant,seed):raise ValueError('Actual shared union training identity differs')
                ts['runs'].append(dict(job,variant=variant,seed=seed,checkpoint=ref(d/'model.pt'),metrics=ref(d/'metrics.json'),run_config=ref(d/'run_config.json'),history=ref(d/'train_history.json'),normalization=ref(d/'normalization.json'),actual_CUDA=True));atomic(OUT/'training/summary.json',ts);update(completed_trainings=len(ts['runs']))
        ts.update(status='COMPLETE',completed_utc=now());atomic(OUT/'training/summary.json',ts);ins=dict(status='RUNNING',protocol=pr,jobs=[],expected_jobs=63,training_summary=ref(OUT/'training/summary.json'),GT_annotations_read=False,AP_computed=False);ens=dict(status='IN_PROGRESS',protocol=pr,records=[],expected_records=21)
        pending=list(P['inference_cases'])
        while pending:
            ready=[]
            for case in pending:
                sp=OUT/'features'/case/'status.json'
                if sp.exists():
                    r=read(sp)
                    if r['status']=='FAILED':raise RuntimeError('Feature failure '+case)
                    if r['status']=='COMPLETE':ready.append(case)
            if not ready:update(status='WAITING_UNION_CASES',pending_cases=pending);time.sleep(2);continue
            case=ready[0];fmp=OUT/'features'/case/'feature_manifest.json';fm=read(fmp);valid=np.load(verify(fm['outputs']['candidate_valid.npy']),allow_pickle=False)
            for variant in VARIANTS:
                values=[];inputs=[]
                for seed in [17,29,43]:
                    update(status='RUNNING_ACTUAL_CUDA_INFERENCE');d=OUT/'inference'/case/variant/f'seed{seed}';cp=OUT/'models'/variant/f'seed{seed}'/'model.pt';job=run(f'infer_{case}_{variant}_{seed}',[sys.executable,'-B',str(CODE/'scripts/infer_union.py'),'--feature-manifest',str(fmp),'--checkpoint',str(cp),'--protocol',str(pp),'--output-dir',str(d),'--device','cuda:0']);rp=d/'inference.json';r=read(rp)
                    if r['protocol_sha256']!=PSHA or(r['case'],r['variant'],r['seed'])!=(case,variant,seed)or r['checkpoint']!=ref(cp)or r['feature_manifest']!=ref(fmp)or r['candidate_row_key_sha256']!=fm['candidate_row_key_sha256']or r['CUDA_VISIBLE_DEVICES']!=GPU_UUID:raise ValueError('Actual union inference identity mismatch')
                    with np.load(verify(r['output']),allow_pickle=False)as z:q=z['quality'].copy()
                    if q.dtype!=np.float32 or q.shape!=valid.shape or np.any(q[~valid]!=0)or not np.isfinite(q).all():raise ValueError('Malformed union seed quality')
                    ins['jobs'].append(dict(job,case=case,variant=variant,seed=seed,receipt=ref(rp),scores=r['output'],actual_CUDA=True));atomic(OUT/'inference/summary.json',ins);values.append(q.astype(np.float64));inputs.append(dict(seed=seed,scores=r['output'],inference_receipt=ref(rp),checkpoint=r['checkpoint']));update(completed_inferences=len(ins['jobs']))
                q=np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32);d=OUT/'ensembles'/case/variant/'all3';d.mkdir(parents=True,exist_ok=True);sp=d/'quality_scores.npz';rp=d/'ensemble.json'
                with sp.open('xb')as f:np.savez_compressed(f,quality=q)
                receipt=dict(status='COMPLETE_SAVED_SHARED_UNION_ENSEMBLE',created_utc=now(),protocol=pr,protocol_sha256=PSHA,case=case,variant=variant,seeds=[17,29,43],inputs=inputs,feature_manifest=ref(fmp),candidate_row_key_sha256=fm['candidate_row_key_sha256'],shape=list(q.shape),dtype='float32',scores=ref(sp),output_sha256=sha(sp),invalid_quality_exact0=True,rule='F64 arithmetic mean of all3 seeds followed by one F32 cast',GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False);atomic(rp,receipt);record=dict(case=case,variant=variant,group='all3',scores=ref(sp),inference_receipt=ref(rp),candidate_row_key_sha256=fm['candidate_row_key_sha256']);ens['records'].append(record);atomic(OUT/'ensemble_index.json',ens);atomic(OUT/'ensembles'/case/variant/'status.json',dict(status='COMPLETE',case=case,variant=variant,protocol=pr,ensemble=record));update(completed_ensembles=len(ens['records']))
            pending.remove(case)
        if len(ins['jobs'])!=63 or len(ens['records'])!=21:raise ValueError('Expected9train/63infer/21means')
        ins.update(status='COMPLETE',completed_utc=now());ens.update(status='COMPLETE',completed_utc=now());atomic(OUT/'inference/summary.json',ins);atomic(OUT/'ensemble_index.json',ens);update(status='COMPLETE',active=None,completed_utc=now());print(json.dumps(dict(status='COMPLETE',trainings=9,inferences=63,ensembles=21)),flush=True)
    except BaseException as e:update(status='FAILED',error=repr(e),completion_claimed=False);raise
if __name__=='__main__':main()
