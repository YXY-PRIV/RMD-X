"""Read-only replay of formal Tavailability deployment, never producer-q aliasing.

The old producer uses chunked, valid-compressed sigmoid. Its deployed endpoint
uses dense CPU F32 sigmoid then exact K-q fallback, so these are separate arrays.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
C = Path(__file__).resolve().parents[1]
T_CODE = C.parent/'material_reliability_20261101'
T_OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_reliability_20261101')
SEEDS = [17, 29, 43]


def read(p):
    return json.loads(Path(p).read_text())


def ref(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''): h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(r):
    p = Path(r['path']).resolve()
    assert ref(p) == {k: r[k] for k in ['path', 'bytes', 'sha256']}, str(p)
    return p


def loadz(p):
    with np.load(p, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def original_endpoint(P):
    path = verify(P['sources']['T_model_source'])
    assert path == T_CODE/'src/quality_model.py'
    source = T_CODE/'src/reliability_endpoint.py'
    # This source is part of the sealed T manifest; it is never modified or run
    # as an actor. Only its pure ensemble/mixing functions are called here.
    manifest = read(verify(P['sources']['T_output_manifest']))
    match = [r for r in manifest['source_code_files'] if r['path'] == str(source)]
    assert len(match) == 1; verify(match[0])
    spec = importlib.util.spec_from_file_location('_conditional_readonly_T_deployment', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module, ref(source)


def rebuild(case, P):
    import torch
    assert torch.get_num_threads() == 8, 'Replay the original deployment CPU thread budget'
    old_protocol = P['sources']['T_protocol']; oldP = read(verify(old_protocol))
    assert oldP['seeds'] == SEEDS and oldP['mixing']['fixed_mix_weight'] == .5
    assert oldP['selection']['alphas'] == [0., .25, .5, 1.]
    module, source = original_endpoint(P)
    c = oldP['cases'][case]
    nf = read(verify(c['N_feature_manifest']))
    M = np.load(verify(nf['outputs']['candidate_valid.npy']), allow_pickle=False)
    Fref = c['C_feature_manifest']; assert c['feature_manifest'] == Fref
    base, _, _, Kseedq = module.legacy_quality(case, oldP, Fref, M)
    seedq, logits, receipts, heads = {}, {}, [], []
    for seed in SEEDS:
        record = P['cases'][case]['Tavailability_inference'][str(seed)]
        receipt = read(verify(record['receipt']))
        assert receipt['protocol'] == old_protocol and (receipt['case'], receipt['variant'], receipt['seed']) == (case, 'availability_control', seed)
        assert receipt['checkpoint'] == P['sources']['T_checkpoints']['availability_control'][str(seed)]
        assert receipt['output'] == record['components'] and receipt['feature_manifest'] == c['N_feature_manifest']
        z = loadz(verify(record['components']))
        for k in ['anchor_logits', 'correction_logits', 'full_logits', 'quality', 'availability']:
            assert z[k].dtype == np.float32 and z[k].shape == M.shape and np.isfinite(z[k]).all() and not np.any(z[k][~M])
        delta = z['correction_logits']; a = z['availability']
        expected = np.where(a > 0., z['anchor_logits']+delta, z['anchor_logits'])
        assert np.array_equal(expected.view(np.uint32), z['full_logits'].view(np.uint32))
        q = torch.sigmoid(torch.from_numpy(z['anchor_logits'])+torch.from_numpy(delta)).numpy().astype(np.float32)
        fallback = (delta == 0.) | (a == 0.) | ~M
        q[fallback] = Kseedq[seed][fallback]
        module.check_quality(q, M)
        seedq[seed] = q; logits[seed] = z['full_logits']; heads.append(q.astype(np.float64))
        receipts.append(dict(seed=seed, receipt=record['receipt'], components=record['components'], checkpoint=receipt['checkpoint'],
            deployment_q_F32_SHA256=hashlib.sha256(q.tobytes()).hexdigest(), saved_anchor_full_logits_F32_SHA256=hashlib.sha256(z['full_logits'].tobytes()).hexdigest(),
            producer_quality_difference_cells=int(np.count_nonzero(q.view(np.uint32) != z['quality'].view(np.uint32))),
            Kquality_fallback_cells=int((fallback & M).sum())))
    head = np.mean(np.stack(heads), axis=0).astype(np.float32)
    mixed = module.mix_visual(base, head, M)
    old_index = read(verify(P['sources']['T_final_run_index']))
    assert old_index['status'] == 'COMPLETE' and old_index['protocol'] == old_protocol
    rows = [r for r in old_index['source_records']+old_index['endpoints'] if r['dataset'] == case and r['variant'] == 'availability_control'
            and r['route'] == 'adaptive_router' and r['residual_alpha'] == 1.]
    assert len(rows) == 1
    archived = rows[0]; export = read(verify(archived['export_receipt']))
    assert export['identity']['variant'] == 'availability_control' and export['identity']['route'] == 'adaptive_router' and export['identity']['residual_alpha'] == 1.
    tp = verify(export['score_trace'])
    with np.load(tp, allow_pickle=False) as archive:
        saved = archive['full_candidate_quality']
        branch = archive['selected_branch'].copy()
        union_index = archive['selected_union_index'].copy()
        keep = archive['selected_union_keep_mask'].copy()
    assert np.array_equal(mixed.view(np.uint32), saved.view(np.uint32)), 'Formal strong anchor differs from archived T endpoint'
    reference = np.full(len(M), -1, np.int8); reference[union_index] = branch
    assert np.array_equal(reference >= 0, keep)
    report = dict(case=case, actual_proposals=len(M), valid_candidate_tuples=int(M.sum()),
        input_protocol=old_protocol, original_endpoint_source=source, seeds=receipts,
        archived_export=archived['export_receipt'], archived_metrics=archived['metrics'], archived_score_trace=export['score_trace'],
        deployed_seed_mean_F32_SHA256=hashlib.sha256(head.tobytes()).hexdigest(),
        mixed_quality_F32_SHA256=hashlib.sha256(mixed.tobytes()).hexdigest(),
        archived_mixed_quality_F32_bitdiff=0, baseline_route='Original T availability_control alpha1 adaptive_router with E3 fixed lambda.5',
        original_boxes_rawscores_negative_UNC_unchanged=True, original_Kgeometry_locked_not_used=True,
        new_training_GPU_GT_AP_or_matching=0)
    return dict(mixed=mixed, base=base, seed_quality=seedq, seed_logits=logits, valid=M,
        reference_selected_branch=reference, archived_export=export, archived_record=archived, report=report)
