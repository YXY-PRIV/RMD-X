"""Read-only legacy candidates with newly trained anchored quality corrections.

The exporter never invents boxes/classes or re-runs the original UNC filter.
Only this experiment's output root may be written. Native COCO uses the original
GT/params/vendor chain; a metric is reused only for an exact full input identity.
"""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_conditional_ranking_fix1_20261101')
NIGHT_CODE = CODE.parent/'night_exploration_20261101'
NIGHT_OUT = OUT.parent/'night_exploration_20261101'
sys.path.insert(0, str(NIGHT_CODE))
ce = importlib.import_module('batch_c_shared_union.src.union_endpoint')
HELPER = CODE.parent/'appearance_quality_reranking_20261031/src/quality_endpoint.py'
GAUSSIAN = CODE.parent/'appearance_candidate_refinement_20261031/src/candidate_endpoint.py'
GSHA = '9d818252fcf47334ea2019c4cb0466b6c6244924418d041b0ee9c9644c863fac'
spec = importlib.util.spec_from_file_location('_conditional_ranking_readonly_coco', HELPER)
pure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pure)
spec = importlib.util.spec_from_file_location('_conditional_ranking_readonly_gaussian', GAUSSIAN)
gauss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gauss)
VARIANTS = ['material', 'permuted_material']
ROUTES = ['anchor_locked', 'adaptive_router']
SEEDS = [17, 29, 43]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def verify(r):
    assert ref(r['path']) == {k: r[k] for k in ('path', 'bytes', 'sha256')}, r['path']
    return Path(r['path'])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT), p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def loadnpz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def frozen(protocol, require_runtime_freeze=False):
    p = Path(protocol).resolve()
    expected = os.environ.get('RAXO_PROTOCOL_SHA256')
    assert p == CODE/'config/protocol.json' and expected and sha(p) == expected
    P = read(p)
    assert P['status'].startswith('FROZEN') and P['variants'] == VARIANTS and P['seeds'] == SEEDS
    assert P['routing']['tau'] == 0. and P['routing']['routes'] == ROUTES
    assert P['selection']['alphas'] == [0., .25, .5, 1.]
    assert P['selection']['weights'] == dict(material=.5, permuted_material=.5)
    assert P['architecture']['total_aux_head_parameters'] == 1717 and P['architecture']['frozen_parameters'] == 1355 and P['architecture']['trainable_parameters'] == 362
    assert P['budget']['new_trainings'] == 6 and P['budget']['new_full_C_GPU_inferences'] == 42
    assert [P['budget'][k] for k in ['source_AP_logical', 'test_AP_logical', 'total_AP_logical']] == [16, 24, 40]
    assert P['training']['validation_loss'] == 'conditional_pairwise_only'
    assert P['mixing']['fixed_mix_weight'] == .5 and P['score']['alpha'] == .5 and P['score']['sigma'] == 1.
    assert sha(GAUSSIAN) == GSHA
    pr = ref(p)
    fp = OUT/'automation/endpoint_code_freeze.json'
    if require_runtime_freeze:
        assert fp.exists(), 'Actual export/evaluation requires the conductor endpoint source freeze'
    if fp.exists():
        f = read(fp)
        assert f['protocol'] == pr
        required_new = {str(CODE/name) for name in [
            'src/conditional_endpoint.py', 'src/anchor_reference.py', 'scripts/evaluate.py', 'scripts/run_evaluations.py']}
        assert required_new <= {r['path'] for r in f['sources']}
        for key in ['sources', 'readonly_sources', 'native_libraries']:
            values = f[key]
            if isinstance(values, dict):
                values = values.values()
            for r in values:
                verify(r)
    return P, pr


@contextlib.contextmanager
def lock(path):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT), p
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source(case, protocol, output_root=OUT):
    assert Path(output_root).resolve() == OUT
    P, pr = frozen(protocol, require_runtime_freeze=True)
    assert case in P['inference_cases']
    cpp = verify(P['sources']['C_protocol'])
    for key in ['K_protocol', 'N_protocol']:
        verify(P['sources'][key])
    _, _, _, F, Fref, a, C, M, mapping, records = ce.source(case, cpp, ce.OUT)
    c = P['cases'][case]
    assert c['feature_manifest'] == c['C_feature_manifest'] == Fref
    assert c['candidate_row_key_sha256'] == F['candidate_row_key_sha256']
    N = read(verify(c['N_feature_manifest']))
    assert N['source_C_feature_manifest'] == Fref
    assert N['candidate_row_key_sha256'] == F['candidate_row_key_sha256']
    assert N['outputs']['candidate_valid.npy'] == F['outputs']['candidate_valid.npy']
    assert N['outputs']['candidate_category_id.npy'] == F['outputs']['candidate_category_id.npy']
    aligned = read(verify(c['auxiliary_feature_manifest']))
    assert aligned['protocol'] == c['auxiliary_protocol']
    assert aligned['outputs']['candidate_valid.npy'] == N['outputs']['candidate_valid.npy']
    assert aligned['outputs']['candidate_category_id.npy'] == N['outputs']['candidate_category_id.npy']
    return P, pr, F, Fref, a, C, M, mapping, records


def check_quality(q, M):
    assert q.dtype == np.float32 and q.shape == M.shape
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)) and not np.any(q[~M])


def mix_visual(base, head, valid):
    q = np.zeros(valid.shape, np.float32)
    b = np.clip(base[valid].astype(np.float64), 1e-6, 1.)
    h = np.clip(head[valid].astype(np.float64), 1e-6, 1.)
    q[valid] = (np.power(b, .5)*np.power(h, .5)).astype(np.float32)
    return q


def legacy_quality(case, P, Fref, M):
    c = P['cases'][case]
    base = loadnpz(verify(c['Ebase_scores']))['quality']
    er = read(verify(c['Ebase_receipt']))
    assert er['case'] == case and er['variant'] == 'visual_clean' and er['seeds'] == SEEDS
    assert er['feature_manifest'] == Fref and er['scores'] == c['Ebase_scores']
    geometry = loadnpz(verify(c['Kgeometry_scores']))['quality']
    gr = read(verify(c['Kgeometry_receipt']))
    assert gr['case'] == case and gr['variant'] == 'geometry' and gr['seeds'] == SEEDS
    assert gr['scores'] == c['Kgeometry_scores']
    assert gr['candidate_row_key_sha256'] == P['cases'][case]['candidate_row_key_sha256']
    check_quality(base, M)
    check_quality(geometry, M)
    seed_inputs = {i['seed']: i for i in gr['inputs']}
    assert set(seed_inputs) == set(SEEDS)
    seedq = {}
    for seed in SEEDS:
        s = seed_inputs[seed]
        sr = read(verify(s['inference_receipt']))
        assert sr['seed'] == seed and sr['case'] == case and sr['variant'] == 'geometry'
        assert sr['checkpoint'] == s['checkpoint'] == P['sources']['K_geometry_checkpoints'][str(seed)]
        seedq[seed] = loadnpz(verify(s['scores']))['quality']
        check_quality(seedq[seed], M)
    independent_mean = np.mean(np.stack([seedq[s].astype(np.float64) for s in SEEDS]), axis=0).astype(np.float32)
    assert np.array_equal(independent_mean.view(np.uint32), geometry.view(np.uint32))
    return base, geometry, mix_visual(base, geometry, M), seedq


def quality(case, variant, alpha, P, pr, Fref, M):
    from src.anchor_reference import rebuild
    assert variant in VARIANTS and alpha in P['selection']['alphas']
    anchor = rebuild(case, P)
    assert np.array_equal(anchor['valid'], M)
    base, q0 = anchor['base'], anchor['mixed']
    c = P['cases'][case]
    nf = read(verify(c['auxiliary_feature_manifest']))
    assert nf['protocol'] == c['auxiliary_protocol'] and nf['source_C_feature_manifest'] == Fref
    assert nf['candidate_row_key_sha256'] == c['candidate_row_key_sha256']
    inputs = []
    if alpha == 0.:
        q = q0.copy()
        head = np.mean(np.stack([anchor['seed_quality'][s].astype(np.float64) for s in SEEDS]), axis=0).astype(np.float32)
    else:
        import torch
        X = np.load(verify(nf['outputs']['candidate_features.npy']), mmap_mode='r', allow_pickle=False)
        ci = np.load(verify(nf['outputs']['candidate_class_order_index.npy']), allow_pickle=False)
        has = np.load(verify(nf['outputs']['actual_supported_class_mask.npy']), allow_pickle=False)
        T = np.load(verify(nf['outputs']['supported_reference_T.npy']), allow_pickle=False)
        qv = np.load(verify(nf['outputs']['query_valid.npy']), allow_pickle=False)
        assert X.dtype == np.float32 and X.shape == M.shape+(13,)
        assert ci.dtype == np.int64 and ci.shape == M.shape and np.all(ci[~M] == -1)
        assert has.dtype == bool and has.ndim == 1 and qv.dtype == bool and qv.shape == (len(M),)
        assert T.dtype == np.int64 and np.array_equal(T, np.where(qv, has.sum(), 0))
        availability = ((X[:, :, 12] == 1.) & M).astype(np.float32)
        idx = np.flatnonzero(has)
        heads = []
        for seed in SEEDS:
            directory = OUT/'inference'/case/variant/f'seed{seed}'
            rp = directory/'inference.json'; r = read(rp)
            assert r['status'] == 'COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE'
            assert r['protocol'] == pr and (r['case'], r['variant'], r['seed']) == (case, variant, seed)
            assert r['source_auxiliary_feature_manifest'] == c['auxiliary_feature_manifest']
            assert r['candidate_row_key_sha256'] == c['candidate_row_key_sha256']
            assert r['Tavailability_inference'] == c['Tavailability_inference'][str(seed)]
            assert verify(r['checkpoint']) == OUT/'models'/variant/f'seed{seed}'/'model.pt'
            assert r['GT_opened'] is False and r['labels_read'] is False and r['AP_computed'] is False
            assert r['actual_CUDA'] is True
            fp = verify(r['feature_manifest']); assert fp == OUT/'features'/case/'feature_manifest.json'
            fm = read(fp); assert fm['protocol'] == pr
            qp = verify(r['output']); assert qp == directory/'components.npz'
            z = loadnpz(qp)
            for key in ['anchor_logits', 'raw_candidate_correction', 'centered_correction', 'availability']:
                assert z[key].dtype == np.float32 and z[key].shape == M.shape and np.isfinite(z[key]).all() and not np.any(z[key][~M])
            assert z['supported_mean'].dtype == np.float64 and z['supported_mean'].shape == (len(M),)
            assert z['gate_weights'].dtype == np.float32 and z['gate_weights'].shape == (len(M),) and np.isfinite(z['gate_weights']).all()
            assert np.all((z['gate_weights'] >= 0.) & (z['gate_weights'] <= 1.))
            bounded = z['candidate_bounded_residual']
            assert bounded.dtype == np.float32 and bounded.shape == M.shape and np.isfinite(bounded).all() and np.all(np.abs(bounded) <= 2.) and not np.any(bounded[~M])
            expected_raw = (z['availability']*z['gate_weights'][:, None])*bounded
            assert np.array_equal(expected_raw.view(np.uint32), z['raw_candidate_correction'].view(np.uint32))
            assert z['valid'].dtype == bool and np.array_equal(z['valid'], M)
            assert np.array_equal(z['availability'], availability) and np.array_equal(z['supported_count'], T)
            assert np.array_equal(z['candidate_class_order_index'], ci)
            assert np.array_equal(z['anchor_logits'].view(np.uint32), anchor['seed_logits'][seed].view(np.uint32))
            raw = np.load(verify(r['full_raw_correction']), mmap_mode='r', allow_pickle=False)
            centered = np.load(verify(r['full_centered_correction']), mmap_mode='r', allow_pickle=False)
            perm = np.load(verify(r['permutation_index']), mmap_mode='r', allow_pickle=False)
            assert raw.dtype == centered.dtype == np.float32 and raw.shape == centered.shape == (len(M), len(has))
            assert perm.dtype == np.int64 and perm.shape == raw.shape
            mu = np.zeros(len(M), np.float64); raw_gather = np.zeros(M.shape, np.float32); center_gather = np.zeros(M.shape, np.float32)
            for lo in range(0, len(M), 8192):
                hi = min(lo+8192, len(M)); block = np.asarray(raw[lo:hi]); observed = np.asarray(centered[lo:hi]); pi = np.asarray(perm[lo:hi])
                assert np.isfinite(block).all() and np.all(np.abs(block) <= 2.) and not np.any(block[:, ~has]) and not np.any(block[~qv[lo:hi]])
                assert np.array_equal(pi[:, ~has], np.broadcast_to(np.flatnonzero(~has), pi[:, ~has].shape))
                assert np.array_equal(np.sort(pi[:, idx], axis=1), np.broadcast_to(idx, pi[:, idx].shape))
                if variant == 'material':
                    assert np.array_equal(pi, np.broadcast_to(np.arange(len(has)), pi.shape))
                mu[lo:hi] = np.divide(block[:, idx].astype(np.float64).sum(axis=1), T[lo:hi], out=np.zeros(hi-lo, np.float64), where=T[lo:hi] > 0)
                formula = np.zeros(block.shape, np.float64)
                many = T[lo:hi] > 1
                formula[:, idx] = (block[:, idx].astype(np.float64)-mu[lo:hi, None])*.5
                formula[~many] = 0.
                assert np.array_equal(formula.astype(np.float32).view(np.uint32), observed.view(np.uint32))
                assert np.all(np.abs(observed) <= 2.) and not np.any(observed[:, ~has])
                if np.any(many):
                    assert np.all(np.abs(formula[many][:, idx].sum(axis=1)/T[lo:hi][many]) <= 1e-12)
                    assert np.all(np.abs(observed[many][:, idx].astype(np.float64).sum(axis=1)) <= T[lo:hi][many]*2.**-23+1e-12)
                for slot in [0, 1]:
                    rows = np.flatnonzero(M[lo:hi, slot]); cols = ci[lo+rows, slot]
                    raw_gather[lo+rows, slot] = block[rows, cols]
                    center_gather[lo+rows, slot] = observed[rows, cols]
            assert np.array_equal(mu.view(np.uint64), z['supported_mean'].view(np.uint64))
            assert np.array_equal(raw_gather.view(np.uint32), z['raw_candidate_correction'].view(np.uint32))
            assert np.array_equal(center_gather.view(np.uint32), z['centered_correction'].view(np.uint32))
            dz = z['centered_correction']; available = (availability > 0.) & (T[:, None] > 1) & M
            assert not np.any(dz[~available])
            new = torch.sigmoid(torch.from_numpy(z['anchor_logits'])+np.float32(alpha)*torch.from_numpy(dz)).numpy().astype(np.float32)
            exact = (dz == 0.) | ~available
            new[exact] = anchor['seed_quality'][seed][exact]
            check_quality(new, M); heads.append(new.astype(np.float64))
            inputs.append(dict(seed=seed, receipt=ref(rp), components=ref(qp), checkpoint=r['checkpoint'],
                full_raw_correction=r['full_raw_correction'], full_centered_correction=r['full_centered_correction'], permutation_index=r['permutation_index'],
                source_Tavailability_components=c['Tavailability_inference'][str(seed)], feature_manifest=r['feature_manifest'], exact_formal_Tquality_fallback_cells=int((exact & M).sum())))
        head = np.mean(np.stack(heads), axis=0).astype(np.float32)
        q = mix_visual(base, head, M)
    check_quality(q, M); check_quality(q0, M); check_quality(head, M)
    assert np.all(q[M] > 0.)
    return q, q0, dict(kind='TRAINED_SUPPORTED_CONDITIONAL_PAIRWISE_MATERIAL_RESIDUAL', variant=variant, residual_alpha=float(alpha),
        formal_strong_anchor=anchor['report'], strong_anchor_export=anchor['archived_record']['export_receipt'],
        strong_anchor_branch_SHA256=hashlib.sha256(anchor['reference_selected_branch'].tobytes()).hexdigest(),
        base_scores=c['Ebase_scores'], new_component_inputs=inputs, auxiliary_feature_manifest=c['auxiliary_feature_manifest'], source_C_feature_manifest=Fref,
        head_mean_sha256=hashlib.sha256(head.tobytes()).hexdigest(), mixed_array_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
        alpha0_exact_formal_Tavailability_mix=alpha == 0., total_composite_parameters_per_seed=1717, frozen_strong_anchor_parameters_per_seed=1355, trainable_branch_parameters_per_seed=362,
        permutation_control_not_equal_effective_capacity_or_causal_counterfactual=True, context_entropy_is_material_not_visual=True,
        precision='Raw fullC GPU F32; canonical supported F64 mean; center/2 F64 toF32; old saved T full_logits plus F32alpha*delta then dense CPU F32 sigmoid; delta0 copies formal T perseed q; F64 seedmean->F32; original E3 F64 mix->F32',
        all_same_class_candidates_share_one_quality=True, old_boxes_scores_negative_UNC_unchanged=True)


def branch_quality(q, kept, mapping):
    effective = np.zeros(kept.shape, np.float32)
    for branch in [0, 1]:
        rows = np.flatnonzero(kept[:, branch])
        effective[rows, branch] = q[rows, mapping[rows, branch]]
    return effective


def choose(kept, effective):
    selected = np.full(len(kept), -1, np.int8)
    selected[kept[:, 0]] = 0
    use_f = kept[:, 1] & ((effective[:, 1].astype(np.float64)-effective[:, 0].astype(np.float64)) > 0.)
    selected[use_f] = 1
    return selected


def legacy_metric_candidates(case, P):
    prior = read(verify(P['sources']['T_final_run_index']))
    assert prior['status'] == 'COMPLETE' and prior['protocol'] == P['sources']['T_protocol']
    archived = [r['metrics'] for r in prior['source_records']+prior['endpoints'] if r['dataset'] == case]
    if case != 'CLC_tune':
        r = P['cases'][case]['Kgeometry_final_router_metrics']
        return ([r] if r is not None else [])+archived
    p = NIGHT_OUT/'batch_k_multisource_quality/parameters/selected_parameters.json'
    old = read(p)
    rows = [r for r in old['candidate_results'] if r['variant'] == 'geometry' and r['mix_weight'] == .5 and r['route'] == 'router']
    assert len(rows) == 1
    return [ref(rows[0]['metrics'])]+archived


def phase_guard(case, phase, alpha, P, pr):
    assert case in P['inference_cases'] and alpha in P['selection']['alphas']
    data = read(OUT/'verification/data_review.json')
    assert data['status'] == 'PASS' and data['protocol_sha256'] == pr['sha256']
    training = read(OUT/'verification/training_review.json')
    assert training['status'] == 'PASS' and training['protocol_sha256'] == pr['sha256']
    if case == 'CLC_tune':
        assert phase == 'calibration', 'CLC_tune is permitted only in the source calibration phase'
        return
    assert case in P['test_cases'] and phase == 'final'
    selection_path = OUT/'parameters/selected_parameters.json'
    selection = read(selection_path)
    assert selection['status'] == 'FROZEN_SHARED_PARAMETERS_BEFORE_ANY_TEST_AP'
    assert selection['protocol'] == pr and selection['selected_alpha'] == alpha
    gate = read(OUT/'verification/selection_review.json')
    assert gate['status'] == 'PASS' and gate['protocol_sha256'] == pr['sha256']
    assert gate['selected_parameters'] == ref(selection_path)


def export_endpoint(case, variant, route, alpha, protocol, output_root=OUT, phase='final'):
    assert variant in VARIANTS and route in ROUTES
    P, pr, F, Fref, a, candidates, valid, mapping, records = source(case, protocol, output_root)
    phase_guard(case, phase, alpha, P, pr)
    q, q0, qr = quality(case, variant, float(alpha), P, pr, Fref, valid)
    eff = branch_quality(q, a['branch_kept'], mapping)
    anchor_eff = branch_quality(q0, a['branch_kept'], mapping)
    reference = choose(a['branch_kept'], anchor_eff)
    assert hashlib.sha256(reference.tobytes()).hexdigest() == qr['strong_anchor_branch_SHA256']
    selected = reference.copy() if route == 'anchor_locked' else choose(a['branch_kept'], eff)
    assert np.array_equal(selected >= 0, reference >= 0), 'Shared positive union membership must remain fixed'
    if route == 'anchor_locked' or alpha == 0.:
        assert np.array_equal(selected, reference)
    same = a['branch_kept'].all(1) & (a['branch_category_id'][:, 0] == a['branch_category_id'][:, 1])
    assert np.array_equal(eff[same, 0], eff[same, 1])
    rows = np.flatnonzero(selected >= 0)
    branch = selected[rows].astype(np.int64)
    slot = mapping[rows, branch]
    category = a['branch_category_id'][rows, branch]
    sq = q[rows, slot]
    raw = a['raw_score'][rows].astype(np.float64)
    clipped = np.clip(sq.astype(np.float64), 1e-6, 1.)
    pre = np.power(raw, .5)*np.power(clipped, .5)
    post, rank = gauss.gaussian(pre, a['bbox'][rows], a['image_id'][rows], category, a['global_cache_index'][rows])
    identity = dict(dataset=case, split='tune' if case == 'CLC_tune' else 'eval', variant=variant, route=route,
        residual_alpha=float(alpha), tau=0., case=phase, protocol=pr, feature_manifest=Fref,
        source_prediction_V=records['V']['prediction'], source_prediction_F=records['F']['prediction'],
        native_source_trace=F['outputs']['source_trace.npz'], quality=qr, exporter=ref(__file__),
        readonly_gaussian=ref(GAUSSIAN), readonly_coco_helpers=ref(HELPER),
        candidate_row_key_sha256=F['candidate_row_key_sha256'], score_alpha=.5, sigma=1.,
        anchor_locked_reference='Archived Tavailability alpha1 adaptive_router +E3 lambda.5 tau0, shared across both roles',
        routing_precision='F32 q toF64; absent quality0; strict F-V>0; equal prefersV')
    key = digest(identity)
    atag = format(float(alpha), '.12g').replace('.', 'p')
    directory = OUT/'endpoints'/case/f'{variant}_{route}_a{atag}_{phase}_{key[:16]}'
    ep = directory/'export.json'
    with lock(OUT/'locks'/('export_'+key+'.lock')):
        if ep.exists():
            e = read(ep)
            assert e['identity_sha256'] == key == digest(e['identity'])
            verify(e['prediction']); verify(e['score_trace'])
            return e
        assert not directory.exists() or not any(directory.iterdir()), directory
        directory.mkdir(parents=True, exist_ok=True)
        pred = directory/'predictions.json'
        save(pred, [dict(image_id=int(a['image_id'][i]), category_id=int(c), bbox=a['bbox'][i].tolist(), score=float(s)) for i, c, s in zip(rows, category, post)])
        if float(alpha) == 0.:
            baseline = read(verify(qr['strong_anchor_export']))
            assert ref(pred)['sha256'] == baseline['prediction']['sha256'], 'Alpha0 must exactly reuse formal Tanchor prediction bytes'
        trace = directory/'score_trace.npz'
        with trace.open('xb') as stream:
            np.savez_compressed(stream, global_cache_index=a['global_cache_index'][rows], selected_union_index=rows,
                selected_branch=branch, selected_candidate_slot=slot, selected_category_id=category,
                selected_native_proto_index=a['branch_native_proto_index'][rows, branch], selected_quality=sq,
                clipped_quality=clipped, native_raw_score=raw, pre_soft_score=pre, post_soft_score=post,
                soft_selection_rank=rank, effective_branch_quality=eff, anchor_branch_quality=anchor_eff,
                full_candidate_quality=q, strong_anchor_reference_candidate_quality=q0,
                strong_anchor_reference_selected_branch=reference, selected_union_keep_mask=selected >= 0)
        counts = dict(union_proposals=len(selected), retained=len(rows), V_selected=int((branch == 0).sum()),
            F_selected=int((branch == 1).sum()), F_only_admitted=int((~a['branch_kept'][rows, 0] & a['branch_kept'][rows, 1]).sum()),
            overlap_F_selected=int((a['branch_kept'][rows, 0] & (branch == 1)).sum()),
            overlap_category_changes=int(((a['branch_category_id'][rows, 0] != category) & a['branch_kept'][rows, 0]).sum()),
            changed_branch_vs_anchor_reference=int((selected != reference).sum()))
        e = dict(status='CONDITIONAL_MATERIAL_ENDPOINT_EXPORTED', created_utc=now(), identity_sha256=key, identity=identity,
            prediction=ref(pred), score_trace=ref(trace), prediction_count=len(rows), counts=counts,
            source_GT=records['V']['GT'], source_params=records['V']['params'],
            native_metric_candidates=[records['V']['metrics_ref'], records['F']['metrics_ref']]+legacy_metric_candidates(case, P),
            source_native_records=records, at_most_one_output_per_global_proposal=True,
            selected_class_from_retained_native_branch=True, selected_branch_original_UNC_passed=True,
            UNC_recomputed=False, new_class_created=False, bbox_rawscore_global_order_preserved=True,
            sameclass_branch_q_exact=True, fixed_union_membership_equals_strong_anchor_reference=True,
            GT_used_for_export=False, anchor_locked_exact_branch_identity=(route == 'anchor_locked'))
        save(ep, e)
        return e


@contextlib.contextmanager
def native_slot(P):
    directory = Path(P['resources']['native_lock_root']).resolve()
    assert directory == OUT/'resources/native_slots' and P['resources']['native_global_slots'] == 4
    directory.mkdir(parents=True, exist_ok=True)
    selected = None
    try:
        while selected is None:
            for i in range(4):
                stream = (directory/f'slot{i}.lock').open('a')
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = stream
                    break
                except BlockingIOError:
                    stream.close()
            if selected is None:
                time.sleep(.2)
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(values['MemAvailable'].strip().split()[0])/2**20
        assert available >= P['resources']['min_MemAvailable_GiB']
        yield
    finally:
        if selected is not None:
            fcntl.flock(selected, fcntl.LOCK_UN)
            selected.close()


def actual(path, prediction_sha, gt_sha, params, sources, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if str(path) in seen:
        return None
    seen.add(str(path))
    m = read(path)
    r = pure._completed_actual_source(path, prediction_sha, gt_sha, params, sources)
    if r is not None:
        return r
    if m.get('AP_reused') and isinstance(m.get('reused_source'), dict) and 'metrics' in m['reused_source']:
        return actual(verify(m['reused_source']['metrics']), prediction_sha, gt_sha, params, sources, seen)
    return None


def evaluate_export(export_receipt, output_root=OUT):
    assert Path(output_root).resolve() == OUT
    ep = Path(export_receipt).resolve()
    assert ep.is_relative_to(OUT)
    e = read(ep)
    assert digest(e['identity']) == e['identity_sha256']
    P, pr = frozen(e['identity']['protocol']['path'], require_runtime_freeze=True)
    assert e['identity']['protocol'] == pr
    assert e['identity']['variant'] in VARIANTS and e['identity']['route'] in ROUTES
    phase_guard(e['identity']['dataset'], e['identity']['case'], e['identity']['residual_alpha'], P, pr)
    verify(e['prediction']); verify(e['score_trace'])
    sources = pure._native_sources()
    mp = ep.parent/'metrics.json'
    if mp.exists():
        m = read(mp)
        assert m['export_sha256'] == sha(ep) and m['identity_sha256'] == e['identity_sha256']
        assert m['prediction'] == e['prediction'] and m['original_GT'] == e['source_GT'] and m['params'] == e['source_params']
        assert all(m[k]['sha256'] == sources[k]['sha256'] for k in sources)
        expected_key = digest(dict(prediction=e['prediction']['sha256'], GT=e['source_GT']['sha256'],
                                   params=e['source_params'], sources={k: v['sha256'] for k, v in sources.items()}))
        assert m['reuse_key'] == expected_key
        assert len(m['stats']) == 12 and np.isfinite(m['stats']).all()
        assert [m['AP'], m['AP50'], m['AP75'], m['AR100']] == [m['stats'][i] for i in [0, 1, 2, 8]]
        assert m['AP_computed_here'] != m['AP_reused']
        verify(m['native_log']); verify(m['original_GT'])
        if m['AP_reused']:
            reused = actual(verify(m['reused_source']['metrics']), e['prediction']['sha256'],
                            e['source_GT']['sha256'], e['source_params'], sources)
            assert reused is not None and read(verify(reused['metrics']))['stats'] == m['stats']
        return m
    gtref = e['source_GT']
    gtpath = verify(gtref)
    gt = read(gtpath)
    ids = [int(a['id']) for a in gt['annotations']]
    assert len(set(ids)) == len(ids) and all(i > 0 for i in ids)
    params = pure._params(gt)
    assert params == e['source_params']
    key = digest(dict(prediction=e['prediction']['sha256'], GT=gtref['sha256'], params=params,
                      sources={k: v['sha256'] for k, v in sources.items()}))
    entry = OUT/'evaluation_registry'/(key+'.json')
    with lock(OUT/'locks'/('evaluation_'+key+'.lock')):
        candidates = []
        if entry.exists():
            candidates.append(verify(read(entry)['actual_metrics']))
        candidates.extend(verify(r) for r in e['native_metric_candidates'])
        reused = None
        for candidate in candidates:
            reused = actual(candidate, e['prediction']['sha256'], gtref['sha256'], params, sources)
            if reused is not None:
                break
        if reused is not None:
            old = read(reused['metrics']['path'])
            m = dict(stats=old['stats'], AP=old['AP'], AP50=old['AP50'], AP75=old['AP75'], AR100=old['stats'][8],
                native_log=old['native_log'], AP_reused=True, AP_computed_here=False,
                actual_AP_computed_here=False, actual_native_evaluation=False, reused_source=reused)
        else:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            log = ep.parent/'native_COCO.log'
            with native_slot(P), log.open('x') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                cg = COCO(str(gtpath))
                if e['prediction_count']:
                    cd = cg.loadRes(e['prediction']['path'])
                else:
                    cd = COCO()
                    cd.dataset = dict(images=gt['images'], categories=gt['categories'], annotations=[])
                    cd.createIndex()
                ev = COCOeval(cg, cd, 'bbox')
                for k in ['imgIds', 'catIds', 'maxDets', 'areaRng', 'areaRngLbl', 'useCats']:
                    setattr(ev.params, k, params[k])
                ev.params.iouThrs = np.asarray(params['iouThrs'], np.float64)
                ev.params.recThrs = np.asarray(params['recThrs'], np.float64)
                ev.evaluate(); ev.accumulate(); ev.summarize()
            stats = [float(x) for x in ev.stats]
            m = dict(stats=stats, AP=stats[0], AP50=stats[1], AP75=stats[2], AR100=stats[8], native_log=ref(log),
                AP_reused=False, AP_computed_here=True, actual_AP_computed_here=True,
                actual_native_evaluation=True, reused_source=None)
        m.update(created_utc=now(), export_sha256=sha(ep), identity_sha256=e['identity_sha256'],
            prediction=e['prediction'], original_GT=gtref, params=params, **sources, reuse_key=key,
            GT_access_purpose='CLC source calibration shared parameters' if e['identity']['dataset'] == 'CLC_tune' else 'Final evaluation only; no reselection',
            native_global_slots=4)
        save(mp, m)
        if not entry.exists():
            save(entry, dict(key=key, actual_metrics=reused['metrics'] if reused is not None else ref(mp)))
        return m
