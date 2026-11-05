#!/usr/bin/env python3
"""Calibrate once, freeze one shared setting, await independent review, test all.

No test AP is evaluated before the shared setting and root selection PASS.
No failed or interrupted controller is automatically retried.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from src.reliability_endpoint import OUT, VARIANTS, ROUTES, frozen, ref, read, save, now


def replace(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+f'.tmp.{os.getpid()}')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temp, p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'calibrate', 'test'], default='all')
    args = parser.parse_args()
    pp = CODE/'config/protocol.json'
    P, pr = frozen(pp)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'automation').mkdir(exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    controller = (OUT/'automation/evaluation_controller.lock').open('a')
    fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_status = OUT/'automation/evaluation_status.json'
    if previous_status.exists():
        previous = read(previous_status)
        assert args.phase == 'test' and previous['status'] == 'CALIBRATION_COMPLETE_AWAITING_INDEPENDENT_SELECTION_REVIEW', (
            'Only the explicit calibration-to-test transition can reuse this controller scope', previous['status'])
    state = dict(status='WAITING_TRAINING_REVIEW', PID=os.getpid(), started_utc=now(), phase=args.phase,
                 active={}, completed_jobs=[])
    mutex = threading.Lock()
    live_children = {}
    beginning = time.monotonic()

    def update(**values):
        with mutex:
            state.update(values, updated_utc=now())
            replace(OUT/'automation/evaluation_status.json', state)

    def gate(name, extra=None):
        p = OUT/'verification'/name
        while not p.exists():
            assert time.monotonic()-beginning < 4*3600, ('Missing conductor gate', name)
            frozen(pp)
            time.sleep(2)
        d = read(p)
        assert d['status'] == 'PASS' and d['protocol_sha256'] == pr['sha256'], (name, d)
        if extra is not None:
            assert d['selected_parameters'] == extra
        return ref(p)

    def job(case, variant, route, alpha, phase):
        assert frozen(pp)[1] == pr
        name = f'{phase}_{case}_{variant}_{route}_a{alpha:g}'
        log = OUT/'logs'/f'eval_{name}.log'
        argv = [sys.executable, '-B', str(CODE/'scripts/evaluate.py'), '--dataset', case,
                '--variant', variant, '--route', route, '--alpha', str(alpha), '--phase', phase, '--protocol', str(pp)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', NUMEXPR_NUM_THREADS='8')
        with log.open('x') as stream:
            proc = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
            with mutex:
                state['active'][name] = dict(PID=proc.pid, argv=argv, log=str(log), started_utc=now())
                live_children[name] = proc
            update()
            code = proc.wait()
        with mutex:
            state['active'].pop(name)
            live_children.pop(name)
            state['completed_jobs'].append(dict(job=name, exit_code=code, completed_utc=now(), log=str(log)))
        update()
        assert code == 0, (name, code, str(log))
        results = []
        for line in log.read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'export_receipt' in value:
                results.append(value)
        assert results
        result = results[-1]
        e, m = read(result['export_receipt']), read(result['metrics'])
        expected = dict(dataset=case, variant=variant, route=route, residual_alpha=alpha, case=phase)
        assert all(e['identity'][k] == v for k, v in expected.items())
        assert e['identity']['protocol'] == pr and m['export_sha256'] == ref(result['export_receipt'])['sha256']
        assert m['AP_computed_here'] != m['AP_reused']
        return dict(**expected, export_receipt=ref(result['export_receipt']), metrics=ref(result['metrics']),
            AP=m['AP'], AP50=m['AP50'], AP75=m['AP75'], AR100=m['stats'][8],
            AP_computed_here=m['AP_computed_here'], AP_reused=m['AP_reused'],
            prediction_count=e['prediction_count'], counts=e['counts'], active_model_files=6)

    def jobs(tasks, collect):
        # All scientific process children are owned by this short-lived pool.
        # The parent service cgroup provides cleanup if the controller fails.
        pending = list(tasks)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = set()
            try:
                while pending or futures:
                    while pending and len(futures) < 4:
                        futures.add(pool.submit(job, *pending.pop(0)))
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.remove(future)
                        collect.append(future.result())
                        replace(OUT/'run_index.json', index)
            except BaseException:
                for future in futures:
                    future.cancel()
                with mutex:
                    owned = list(live_children.values())
                for proc in owned:
                    if proc.poll() is None:
                        proc.terminate()
                raise

    index = None
    try:
        update()
        training = gate('training_review.json')
        if args.phase in ['all', 'calibrate']:
            assert not (OUT/'parameters/selected_parameters.json').exists()
            assert not (OUT/'calibration_index.json').exists()
            index = dict(status='CALIBRATING_SOURCE_ONLY', protocol=pr, training_review=training,
                         source_records=[], endpoints=[])
            update(status='SOURCE_CALIBRATION_24_LOGICAL_SETTINGS')
            tasks = [(P['selection']['dataset'], v, route, float(alpha), 'calibration')
                     for alpha in P['selection']['alphas'] for route in ROUTES for v in VARIANTS]
            jobs(tasks, index['source_records'])
            assert len(index['source_records']) == 24
            calibration = dict(status='COMPLETE_SOURCE_CALIBRATION', completed_utc=now(), protocol=pr,
                               source_records=index['source_records'], test_AP_evaluated=False)
            save(OUT/'calibration_index.json', calibration)
            scores = []
            for alpha in P['selection']['alphas']:
                for route in ROUTES:
                    matching = [r for r in index['source_records'] if r['residual_alpha'] == alpha and r['route'] == route]
                    by_variant = {r['variant']: r for r in matching}
                    assert set(by_variant) == set(VARIANTS) and len(matching) == 3
                    objective = sum(P['selection']['weights'][v]*by_variant[v]['AP'] for v in VARIANTS)
                    scores.append(dict(alpha=float(alpha), route=route, objective=objective,
                                       variant_AP={v: by_variant[v]['AP'] for v in VARIANTS}, records=matching))
            winner = min(scores, key=lambda r: (-r['objective'], r['alpha'], ROUTES.index(r['route'])))
            fixed = dict(status='FROZEN_SHARED_PARAMETERS_BEFORE_ANY_TEST_AP', created_utc=now(), protocol=pr,
                selected_alpha=winner['alpha'], selected_route=winner['route'], selected_objective=winner['objective'],
                source_calibration=ref(OUT/'calibration_index.json'), criterion=P['selection'], all_settings=scores,
                primary_variant=P['primary_variant'], secondary_variant=P['secondary_variant'],
                test_AP_used_for_selection=False, shared_setting_all_variants=True,
                other_route_at_same_alpha_also_reported=True)
            save(OUT/'parameters/selected_parameters.json', fixed)
            index.update(status='CALIBRATION_COMPLETE_TEST_NOT_STARTED', selection=ref(OUT/'parameters/selected_parameters.json'))
            replace(OUT/'run_index.json', index)
            update(status='CALIBRATION_COMPLETE_AWAITING_INDEPENDENT_SELECTION_REVIEW')
            if args.phase == 'calibrate':
                return
        else:
            index = read(OUT/'run_index.json')
            assert index['status'] == 'CALIBRATION_COMPLETE_TEST_NOT_STARTED' and index['protocol'] == pr
            assert index['endpoints'] == []
        selected_path = OUT/'parameters/selected_parameters.json'
        selected_ref = ref(selected_path)
        selected = read(selected_path)
        assert selected['status'] == 'FROZEN_SHARED_PARAMETERS_BEFORE_ANY_TEST_AP'
        assert selected['protocol'] == pr and selected['test_AP_used_for_selection'] is False
        selection_review = gate('selection_review.json', selected_ref)
        assert ref(selected_path) == selected_ref
        index.update(status='TESTING_ALL_PREDECLARED_VARIANTS_AND_ROUTES', selection_review=selection_review)
        replace(OUT/'run_index.json', index)
        update(status='FIXED_SIX_DATASET_TESTS_36_LOGICAL_ENDPOINTS')
        tasks = [(case, v, route, selected['selected_alpha'], 'final')
                 for case in P['test_cases'] for route in ROUTES for v in VARIANTS]
        jobs(tasks, index['endpoints'])
        assert len(index['endpoints']) == 36 and ref(selected_path) == selected_ref
        assert frozen(pp)[1] == pr
        logical = index['source_records']+index['endpoints']
        index.update(status='COMPLETE', completed_utc=now(), budget=dict(
            logical_AP=len(logical), source_logical_AP=24, test_logical_AP=36,
            actual_new_native_AP=sum(r['AP_computed_here'] for r in logical),
            strict_reused_AP=sum(r['AP_reused'] for r in logical)))
        replace(OUT/'run_index.json', index)
        update(status='COMPLETE', completed_utc=now(), budget=index['budget'])
    except BaseException as exc:
        update(status='FAILED', error=repr(exc), automatic_retry_allowed=False)
        raise


if __name__ == '__main__':
    main()
