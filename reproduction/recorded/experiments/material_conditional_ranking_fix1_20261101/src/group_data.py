"""Group the existing aligned source teachers without image, GT or model execution.

Every actual source catalogue class remains in the full-source population.
The conditional subset contains supported positive-versus-zero alternatives;
background and unsupported-positive proposals remain available to regularization.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_conditional_ranking_fix1_20261101')
SCHEMA = 'conditional_grouped_allclass_v1'
SOURCE_NAMES = ('CLC', 'PID', 'Hi')
MAX_CLASSES = 12
GATE_CONTEXT_NAMES = [
    'confidence_percentile', 'clip_log_aspect_over4', 'width_over_image_width',
    'height_over_image_height', 'centerx_over_width', 'centery_over_height',
    'max_other_proposal_IoU', 'sqrt_relative_box_area',
    'supported_B_top1', 'supported_B_top_minus_second_over2',
    'supported_B_softmax_entropy_over_logT', 'log_supported_count_over_log307',
    'query_has_supported_reference',
]
EXPECTED = {
    'train': dict(proposals=17070, eligible=2191, positive_zero_pairs=16776),
    'val': dict(proposals=4018, eligible=493, positive_zero_pairs=3741),
}


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(value):
    assert ref(value['path']) == {k: value[k] for k in ['path', 'bytes', 'sha256']}, value['path']
    return Path(value['path'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def loadz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def read_inputs(P):
    s = P['sources']
    r = read(verify(s['aligned_training_receipt']))
    qa = read(verify(s['aligned_data_review']))
    assert r['status'] == 'COMPLETE' and qa['status'] == 'PASS'
    assert r['training_npz'] == qa['training'] == s['aligned_training']
    assert r['protocol'] == qa['protocol'] and qa['protocol_sha256'] == r['protocol']['sha256']
    catalogue = read(verify(s['N_training_receipt']))
    assert catalogue['status'] == 'COMPLETE'
    data = loadz(verify(s['aligned_training']))
    assert data['feature_names'].astype(str).tolist() == catalogue['feature_names']
    meta = json.loads(str(data['metadata_json'].item()))
    assert meta['protocol_sha256'] == r['protocol']['sha256']
    assert meta['source_N_training'] == r['source_N_training']
    assert catalogue['training_npz'] == r['source_N_training']
    classes = catalogue['source_class_order']
    assert set(classes) == set(SOURCE_NAMES)
    assert [len(classes[n]) for n in SOURCE_NAMES] == [12, 12, 7]
    for name in SOURCE_NAMES:
        assert [c['class_index'] for c in classes[name]] == list(range(len(classes[name])))
        assert len({int(c['category_id']) for c in classes[name]}) == len(classes[name])
    assert classes['CLC'][0]['category_id'] == 0, 'CID0 is a valid class, never padding'
    return data, meta, classes


def index_fold(data, metadata, classes, fold):
    """Inspect class-complete key groups; do not materialize a grouped dataset."""
    arrays = {k: data[k+'_'+fold] for k in [
        'X', 'y', 'source_index', 'image_index', 'global_cache_index',
        'category_id', 'class_count', 'query_valid']}
    n = len(arrays['y'])
    assert arrays['X'].dtype == np.float32 and arrays['X'].shape == (n, 13)
    assert arrays['y'].dtype == np.float32 and arrays['y'].shape == (n,)
    assert np.isfinite(arrays['X']).all() and np.isfinite(arrays['y']).all()
    assert np.all((arrays['y'] == 0.) | ((arrays['y'] >= .5) & (arrays['y'] <= 1.)))
    for name in ['source_index', 'image_index', 'global_cache_index', 'category_id', 'class_count']:
        assert arrays[name].dtype == np.int64 and arrays[name].shape == (n,)
    assert arrays['query_valid'].dtype == bool and arrays['query_valid'].shape == (n,)
    sid, image, cache, cid = [arrays[k] for k in ['source_index', 'image_index', 'global_cache_index', 'category_id']]
    assert set(np.unique(sid)) == {0, 1, 2} and np.all(cache >= 0)
    keys = np.stack([sid, image, cache], axis=1)
    groups, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    assert np.array_equal(counts[inverse], arrays['class_count'])
    assert len(np.unique(np.stack([sid, image, cache, cid], axis=1), axis=0)) == n
    col = np.empty(n, np.int64)
    for source, name in enumerate(SOURCE_NAMES):
        lookup = {int(c['category_id']): int(c['class_index']) for c in classes[name]}
        rows = np.flatnonzero(sid == source)
        assert np.all(arrays['class_count'][rows] == len(classes[name]))
        col[rows] = np.asarray([lookup[int(c)] for c in cid[rows]], np.int64)
    mapping = np.full((len(groups), MAX_CLASSES), -1, np.int64)
    mapping[inverse, col] = np.arange(n, dtype=np.int64)
    catalogue_counts = np.asarray([len(classes[SOURCE_NAMES[int(s)]]) for s in groups[:, 0]], np.int64)
    class_valid = np.arange(MAX_CLASSES)[None, :] < catalogue_counts[:, None]
    assert np.array_equal(mapping >= 0, class_valid)
    first = mapping[:, 0]; qv = arrays['query_valid'][first]
    assert np.array_equal(arrays['query_valid'], qv[inverse])
    assert bits(arrays['X'][:, :8], arrays['X'][first[inverse], :8])
    identities = metadata['image_identities']
    assert all(identities[int(i)]['index'] == int(i) and identities[int(i)]['source'] == SOURCE_NAMES[int(s)] and identities[int(i)]['role'] == fold for s, i, _ in groups)
    assert len(groups) == EXPECTED[fold]['proposals']
    return arrays, groups, mapping, class_valid, catalogue_counts, qv


def population_weights(source, image, eligible):
    """Global weights: equal source, then its eligible images, then proposals."""
    out = np.zeros(len(source), np.float64); statistics = {}
    for sid, name in enumerate(SOURCE_NAMES):
        rows = np.flatnonzero((source == sid) & eligible)
        images, inverse, counts = np.unique(image[rows], return_inverse=True, return_counts=True)
        assert len(images) > 0 and len(rows) > 0
        out[rows] = 1. / (3. * len(images) * counts[inverse].astype(np.float64))
        assert abs(out[rows].sum() - 1./3.) < 1e-12
        statistics[name] = dict(proposals=len(rows), images=len(images),
            source_weight_sum=float(out[rows].sum()), F64_weights_SHA256=digest(out[rows]))
    assert abs(out.sum() - 1.) < 1e-12 and not np.any(out[~eligible])
    return out, statistics


def gate_context(X, support, query_valid):
    """F64 supported bag stats in original canonical class order, finally F32.

    The saved context is shared unchanged by real and permuted branches. No
    entropy/context recomputation after permutation changes summation order.
    """
    assert X.ndim == 3 and X.shape[2] == 13 and X.shape[1] >= 1
    assert support.dtype == bool and support.shape == X.shape[:2]
    out = np.zeros((len(X), 13), np.float64)
    out[:, :8] = X[:, 0, :8].astype(np.float64)
    T = support.sum(1).astype(np.int64)
    for i in range(len(X)):
        t = int(T[i])
        if t == 0:
            continue
        values = X[i, np.flatnonzero(support[i]), 8].astype(np.float64)
        top = float(np.max(values))
        out[i, 8] = top
        if t >= 2:
            second = float(np.partition(values, -2)[-2])
            out[i, 9] = (top - second) / 2.
            exponential = np.exp(values - top)
            probabilities = exponential / np.sum(exponential, dtype=np.float64)
            entropy = -np.sum(probabilities * np.log(probabilities), dtype=np.float64)
            out[i, 10] = entropy / np.log(float(t))
        out[i, 11] = np.log(float(max(t, 1))) / np.log(307.)
        out[i, 12] = float(bool(query_valid[i]) and t > 0)
    assert np.isfinite(out).all() and np.all(out[:, 10] >= -1e-12) and np.all(out[:, 10] <= 1.+1e-12)
    return out.astype(np.float32), T


def group_fold(data, metadata, classes, fold):
    arrays, groups, mapping, valid, counts, qv = index_fold(data, metadata, classes, fold)
    n = len(groups)
    X = np.zeros((n, MAX_CLASSES, 13), np.float32); y = np.zeros((n, MAX_CLASSES), np.float32)
    X[valid] = arrays['X'][mapping[valid]]; y[valid] = arrays['y'][mapping[valid]]
    category_id = np.full(valid.shape, -1, np.int64)
    class_index = np.full(valid.shape, -1, np.int64)
    category_id[valid] = arrays['category_id'][mapping[valid]]
    class_index[valid] = np.broadcast_to(np.arange(MAX_CLASSES, dtype=np.int64), valid.shape)[valid]
    assert bits(X[valid], arrays['X'][mapping[valid]]) and bits(y[valid], arrays['y'][mapping[valid]])
    assert not np.any(X[~valid]) and not np.any(y[~valid])
    support = valid & (X[:, :, 12] == 1.) & qv[:, None]
    assert np.all(X[:, :, 12][valid] == support[valid].astype(np.float32))
    assert not np.any(X[:, :, 8:][valid & ~support])
    positive = (y > 0.) & support
    negative = (y == 0.) & support
    npositive, nnegative = positive.sum(1), negative.sum(1)
    T = support.sum(1).astype(np.int64)
    eligible = (npositive > 0) & (nnegative > 0) & (T >= 2)
    assert int(eligible.sum()) == EXPECTED[fold]['eligible']
    assert int((npositive*nnegative).sum()) == EXPECTED[fold]['positive_zero_pairs']
    conditional_weight, conditional_populations = population_weights(groups[:, 0], groups[:, 1], eligible)
    full_weight, full_populations = population_weights(groups[:, 0], groups[:, 1], np.ones(n, bool))
    context, context_T = gate_context(X, support, qv)
    assert np.array_equal(T, context_T)
    result = dict(X=X, y=y, class_valid=valid, support=support,
        source_index=groups[:, 0], image_index=groups[:, 1], global_cache_index=groups[:, 2],
        class_count=counts, category_id=category_id, class_order_index=class_index,
        query_valid=qv, source_row_index=mapping, eligible=eligible,
        conditional_weight=conditional_weight, regularizer_weight=full_weight,
        supported_count=T, gate_context=context)
    summary = dict(fold=fold, full_proposals=n, original_tuples=int(valid.sum()),
        eligible_proposals=int(eligible.sum()), supported_positive_zero_pairs=int((npositive*nnegative).sum()),
        conditional_populations=conditional_populations, full_regularizer_populations=full_populations,
        proposal_key_order='lexicographic(source_index,image_index,global_cache_index)',
        class_order='original per-source canonical class_index; CID0 valid; padding id/index/row=-1',
        source_row_index_SHA256=digest(mapping), all_source_tuples_gathered_exactly_once=True,
        all_source_X_y_F32_bits_preserved=True, full_background_and_unsupported_positive_proposals_retained=True,
        gate_context_F32_SHA256=digest(context), positive_pairs_do_not_treat_other_positive_classes_as_negative=True)
    return result, summary


def inspect_only(P):
    """Source-schema/key inspection only: no new grouped tensors or files."""
    data, metadata, classes = read_inputs(P)
    counts = {}
    for fold in ['train', 'val']:
        arrays, groups, mapping, valid, C, qv = index_fold(data, metadata, classes, fold)
        counts[fold] = dict(source_tuples=len(arrays['y']), proposals=len(groups),
            classes_by_source=[len(classes[n]) for n in SOURCE_NAMES], query_invalid=int((~qv).sum()),
            complete_group_join=True, geometry_same_all_classes=True,
            materialized_new_grouped_X_y_or_context=False)
    return dict(status='PASS_READ_ONLY_SOURCE_SCHEMA_AND_KEY_INSPECTION', schema_version=SCHEMA,
        sources={k: P['sources'][k] for k in ['aligned_training', 'aligned_training_receipt', 'aligned_data_review', 'N_training_receipt']},
        counts=counts, new_files_written=0, new_grouped_data_reconstruction=0,
        new_training_GPU_model_forward_AP_GT_RGB=0)


def save(path, value):
    p = Path(path).resolve(); assert p.is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def prepare(protocol_path, expected_sha256):
    p = Path(protocol_path).resolve(); assert p == CODE/'config/protocol.json'
    pr = ref(p); assert pr['sha256'] == expected_sha256
    P = read(p); assert P['status'].startswith('FROZEN')
    assert P['grouped_data']['schema_version'] == SCHEMA
    freeze = read(OUT/'automation/data_code_freeze.json'); assert freeze['protocol'] == pr
    required = {str(CODE/n) for n in ['src/group_data.py', 'scripts/prepare_data.py']}
    assert required <= {r['path'] for r in freeze['sources']}
    for key in ['sources', 'readonly_sources']:
        for value in freeze[key]: verify(value)
    gate = read(OUT/'verification/data_preflight_review.json')
    assert gate['status'] == 'PASS' and gate['protocol'] == pr and gate['protocol_sha256'] == pr['sha256']
    assert not (OUT/'data_completion.json').exists(), 'No implicit data reconstruction retry'
    data, metadata, classes = read_inputs(P)
    arrays, summaries = {}, []
    for fold in ['train', 'val']:
        grouped, summary = group_fold(data, metadata, classes, fold)
        arrays.update({key+'_'+fold: value for key, value in grouped.items()})
        summaries.append(summary)
    assert not(set(zip(arrays['source_index_train'], arrays['image_index_train'])) &
               set(zip(arrays['source_index_val'], arrays['image_index_val'])))
    lineage = dict(schema_version=SCHEMA, protocol=pr,
        source_aligned_training=P['sources']['aligned_training'],
        source_aligned_training_receipt=P['sources']['aligned_training_receipt'],
        source_aligned_data_review=P['sources']['aligned_data_review'],
        source_class_catalogue=P['sources']['N_training_receipt'],
        image_identities=metadata['image_identities'], source_class_order=classes,
        source_names=list(SOURCE_NAMES), gate_context_names=GATE_CONTEXT_NAMES,
        teacher='Per-class sameclass maxIoU>=.5 retained else0, F32; multiple positives preserved.',
        conditional_eligible='T>=2 and at least one supported y>0 and at least one supported y==0',
        conditional_weights='equal source, eligible image, eligible proposal',
        regularizer_weights='equal source, all source images, all source proposals',
        gate_context_precision='CPU F64 B from original canonical supported class order, entropy temp1, final F32; shared pre-permutation context',
        source_arrays_unchanged=True, new_GT_RGB_encoder_teacher_training_GPU_AP=0)
    arrays['metadata_json'] = np.asarray(json.dumps(lineage, ensure_ascii=False, sort_keys=True))
    arrays['feature_names'] = data['feature_names'].copy()
    arrays['gate_context_names'] = np.asarray(GATE_CONTEXT_NAMES)
    target = OUT/'training/grouped_train_val.npz'; target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream: np.savez_compressed(stream, **arrays)
    receipt = dict(status='COMPLETE_GROUPED_SOURCE_DATA', completed_utc=now(), protocol=pr,
        schema_version=SCHEMA, training_npz=ref(target), source_aligned_training=P['sources']['aligned_training'],
        source_aligned_training_receipt=P['sources']['aligned_training_receipt'],
        source_class_catalogue=P['sources']['N_training_receipt'], feature_names=data['feature_names'].astype(str).tolist(),
        gate_context_names=GATE_CONTEXT_NAMES, source_class_order=classes, folds=summaries,
        regularizer_population_keeps_all_proposals=True, no_hard_onehot_or_positive_class_discard=True,
        material_and_permutation_read_same_original_bag_context=True,
        actual_permutation_and_training_not_executed_in_data_stage=True,
        new_GT_RGB_encoder_teacher_training_GPU_AP=0)
    rp = OUT/'training/data_receipt.json'; save(rp, receipt)
    save(OUT/'data_completion.json', dict(status='COMPLETE_CONDITIONAL_GROUPED_SOURCE_DATA_READY',
        completed_utc=now(), protocol=pr, training=ref(rp), training_npz=ref(target),
        input_aligned_data_review=P['sources']['aligned_data_review'], new_training_GPU_AP=0,
        publication_edits=False))
    return read(OUT/'data_completion.json')
