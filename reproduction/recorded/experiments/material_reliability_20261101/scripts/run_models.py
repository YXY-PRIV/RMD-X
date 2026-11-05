#!/usr/bin/env python3
"""Run only the registered nine CUDA heads and 63 label-free forwards."""
from pathlib import Path
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
C = Path(__file__).resolve().parents[1]
O = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_reliability_20261101')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(p):
    return json.loads(Path(p).read_text())

def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def new(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(d, f, indent=2, allow_nan=False); f.write('\n')

def atomic(p, d):
    q = p.with_name(p.name + f'.tmp.{os.getpid()}')
    new(q, d); os.replace(q, p)

def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    raise AssertionError('MemAvailable missing')

def identity(pid):
    b = Path('/proc') / str(pid)
    s = (b/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(PID=pid, starttime_ticks=int(s[19]),
                argv=[v.decode() for v in (b/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])

def main():
    p = C/'config/protocol.json'; P = read(p); pr = ref(p)
    assert pr['sha256'] == os.environ['RAXO_PROTOCOL_SHA256']
    review = O/'verification/preflight_peer_review.json'
    rr = read(review)
    assert rr['status'].startswith('PASS') and rr['protocol'] == pr
    assert set(rr['authorized_scope']) == {'nine_predefined_GPU_quality_head_trainings',
                                         'sixtythree_label_free_GPU_shared_union_inferences'}
    assert rr['endpoint_not_authorized'] is True
    assert P['variants'] == ['availability_control', 'material', 'ordinal_material']
    assert P['seeds'] == [17, 29, 43] and len(P['inference_cases']) == 7
    lock = (O/'automation/model_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [ref(C/name) for name in ['scripts/run_models.py', 'scripts/train_models.py',
               'src/quality_model.py', 'config/model_design.json', 'config/protocol.json']]
    new(O/'automation/model_code_freeze.json', dict(status='FROZEN_BEFORE_MODEL_EXECUTION',
        created_utc=now(), protocol=pr, sources=sources, preflight_review=ref(review)))
    state = dict(status='STARTING', PID=os.getpid(), started_utc=now(), protocol=pr,
                 active=None, completed_trainings=0, completed_inferences=0)
    def update(**kw):
        state.update(kw, updated_utc=now()); atomic(O/'automation/model_status.json', state)
    def guard():
        assert all(ref(s['path']) == s for s in sources)
        assert available_ram() >= 64
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=P['resources']['GPU_UUID'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8')
    def run(name, args):
        guard()
        while True:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                    '--format=csv,noheader,nounits'], text=True)
            busy = [line for line in apps.splitlines()
                    if line.split(',')[0].strip() == P['resources']['GPU_UUID']]
            if not busy:
                break
            update(status='WAITING_RESERVED_GPU_FREE', busy=busy); time.sleep(10)
        argv = [sys.executable, '-B', str(C/'scripts/train_models.py'), *args,
                '--protocol', str(p), '--protocol-sha256', pr['sha256'], '--device', 'cuda:0']
        log = O/'logs'/f'{name}.log'
        with log.open('x') as f:
            child = subprocess.Popen(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
            ident = identity(child.pid)
            launch = dict(identity=ident, argv=argv, started_utc=now(), log=str(log),
                          GPU_UUID=P['resources']['GPU_UUID'], protocol=pr,
                          available_RAM_GiB=available_ram())
            new(O/'automation'/f'launch_{name}.json', launch)
            update(status='ACTUAL_CUDA_' + args[0].upper(), active=launch)
            code = child.wait()
        assert code == 0, (name, code, str(log))
        update(active=None)
        return dict(**launch, exit_code=code, exited_utc=now())
    try:
        update()
        training = []
        for variant in P['variants']:
            for seed in P['seeds']:
                directory = O/'models'/variant/f'seed{seed}'
                job = run(f'train_{variant}_{seed}', ['train', '--variant', variant,
                    '--seed', str(seed), '--output-dir', str(directory)])
                m = read(directory/'metrics.json')
                assert m['status'] == 'COMPLETE' and m['actual_CUDA']
                training.append(dict(variant=variant, seed=seed, job=job,
                    checkpoint=ref(directory/'model.pt'), metrics=ref(directory/'metrics.json')))
                update(completed_trainings=len(training))
        new(O/'automation/training_complete.json', dict(status='COMPLETE_NINE_ACTUAL_CUDA_TRAININGS',
            completed_utc=now(), protocol=pr, runs=training, new_trainings=9))
        inference = []
        for case in P['inference_cases']:
            for variant in P['variants']:
                for seed in P['seeds']:
                    directory = O/'inference'/case/variant/f'seed{seed}'
                    job = run(f'infer_{case}_{variant}_{seed}', ['infer', '--case', case,
                        '--checkpoint', str(O/'models'/variant/f'seed{seed}/model.pt'),
                        '--output-dir', str(directory)])
                    receipt = directory/'inference.json'; r = read(receipt)
                    assert r['protocol'] == pr and r['case'] == case
                    inference.append(dict(case=case, variant=variant, seed=seed, job=job,
                                          receipt=ref(receipt)))
                    update(completed_inferences=len(inference))
        new(O/'automation/inference_complete.json', dict(status='COMPLETE_63_ACTUAL_CUDA_FORWARDS',
            completed_utc=now(), protocol=pr, jobs=inference, new_GPU_inferences=63,
            no_GT_annotation_file_access=True, no_native_AP_or_bootstrap=True))
        update(status='COMPLETE_MODELS_AND_LABEL_FREE_INFERENCE', active=None, completed_utc=now())
    except BaseException as exc:
        update(status='FAILED', error=repr(exc), automatic_retry_allowed=False)
        raise

if __name__ == '__main__':
    main()
