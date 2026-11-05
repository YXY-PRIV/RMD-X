"""Cached exact22 -> aligned auxiliary CLC material inputs; no image/GT reads.

Only source CLC and CLC_tune columns8..11 change. Original labels, identities,
geometry, availability, PID/Hi and every six-test feature file remain unchanged.
"""
from __future__ import annotations
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_bank_alignment_20261101')
ROOT = CODE.parents[1]
sys.path.insert(0, str(ROOT))
from rmdx import material, preargmax_fusion as producer


def read(p):
    return json.loads(Path(p).read_text())


def ref(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(8 << 20), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(r):
    assert ref(r['path']) == {k: r[k] for k in ['path', 'bytes', 'sha256']}, r['path']
    return Path(r['path'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def writable(p):
    p = Path(p).resolve(); assert p.is_relative_to(OUT); p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save(p, value):
    with writable(p).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def savez(p, **values):
    with writable(p).open('xb') as stream:
        np.savez_compressed(stream, **values)


def array(p, value):
    with writable(p).open('xb') as stream:
        np.save(stream, value, allow_pickle=False)


def loadz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def frozen(path, expected):
    path = Path(path).resolve(); assert path == CODE/'config/protocol.json' and ref(path)['sha256'] == expected
    P, pr = read(path), ref(path)
    assert P['status'].startswith('FROZEN')
    assert P['variants'] == ['availability_control', 'material', 'legacy_material']
    assert P['training']['variants'] == ['material'] and P['seeds'] == [17, 29, 43]
    assert P['alignment']['changed_coordinates'] == [8, 9, 10, 11]
    assert read(verify(P['sources']['old_full_material_replay']))['status'] == 'PASS_ALL_CLC_SOURCE_AND_CT_OLD_BANK_FULL_MATERIAL_REPLAY'
    assert read(verify(P['sources']['T_completion']))['status'].startswith('COMPLETE')
    if P['sources'].get('centered_projection_completion') is not None:
        assert read(verify(P['sources']['centered_projection_completion']))['status'].startswith('COMPLETE')
    freeze = read(OUT/'automation/data_code_freeze.json'); assert freeze['protocol'] == pr
    for key in ['sources', 'readonly_sources']:
        for r in freeze[key]: verify(r)
    gate = read(OUT/'verification/data_preflight_review.json')
    assert gate['status'].startswith('PASS') and gate['protocol'] == pr
    return P, pr


def original_supported(P):
    spec = importlib.util.spec_from_file_location('_alignment_readonly_N_supported', verify(P['sources']['N_supported_source']))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def lookup(P, role):
    reference = read(verify(P['sources']['runtime_lookup_reference']))
    names = [r['category_name'] for r in reference['class_order']]
    entry = reference['records'][role]
    bank = read(verify(entry['bank']))
    # Current producer rebuilds stats from these12 mapped class means; it does
    # not use the bank JSON's full307 bank_feature_stats as the scoring stats.
    actual = producer.build_category_material_matrix(names, names, bank)
    matrix = np.asarray([actual['category_features'][n] for n in names], np.float64)
    mean, std = [np.asarray(actual['stats'][k], np.float64) for k in ['mean', 'std']]
    assert hashlib.sha256(matrix.tobytes()).hexdigest() == entry['class_features_F64_SHA256']
    assert hashlib.sha256(mean.tobytes()).hexdigest() == entry['runtime_mean_F64_SHA256']
    assert hashlib.sha256(std.tobytes()).hexdigest() == entry['runtime_std_F64_SHA256']
    assert entry['all_actual_support_positive'] and len(names) == 12
    return actual, reference['class_order']


def material_columns(raw, lk, supported):
    names = [r['category_name'] for r in supported['class_order']]
    prototype = [material.zscore_feature(lk['category_features'][n], lk['stats']) for n in names]
    B = np.empty((len(raw), 12), np.float32)
    assert raw.dtype == np.float64 and raw.shape[1] == 22 and np.isfinite(raw).all()
    for i, row in enumerate(raw):
        z = material.zscore_feature(row, lk['stats'])
        B[i] = np.asarray([material.cosine_similarity(z, p) for p in prototype], np.float32)
    rank, T = supported['module'].supported_columns(B, np.ones(B.shape, bool))
    assert np.all(T == 12)
    return B, np.concatenate([B[:, :, None], rank], axis=2)


def masked(X, normalization):
    value = X.copy()
    value[..., [k for k in range(13) if k not in normalization['active_coordinates']]] = 0.
    if normalization['constant_coordinates']:
        value[..., normalization['constant_coordinates']] = 0.
    return value


def prepare(path, expected):
    P, pr = frozen(path, expected)
    assert not (OUT/'data_completion.json').exists(), 'No implicit replay of data scope'
    old_ref = P['sources']['N_training']; old = loadz(verify(old_ref))
    new = {k: v.copy() for k, v in old.items()}
    metadata = json.loads(str(old['metadata_json'].item())); identities = metadata['image_identities']
    raw = np.load(verify(P['sources']['CLC_source_exact22']), mmap_mode='r', allow_pickle=False)
    proposal = read(verify(P['sources']['CLC_proposal_meta']))
    assert raw.dtype == np.float64 and raw.shape == (len(proposal), 22)
    assert all(r['cache_index'] == i for i, r in enumerate(proposal))
    old_lookup, classes = lookup(P, 'old_local')
    aligned_lookup, other_classes = lookup(P, 'aligned_transfer')
    assert classes == other_classes
    support = dict(module=original_supported(P), class_order=classes)
    cid_lookup = {int(c['category_id']): int(c['class_index']) for c in classes}
    assert set(cid_lookup) == set(range(12))
    fold_proofs = []
    for fold in ['train', 'val']:
        rows = np.flatnonzero(old['source_index_'+fold] == 0)
        g = old['global_cache_index_'+fold][rows]; im = old['image_index_'+fold][rows]; cid = old['category_id_'+fold][rows]
        unique, inv = np.unique(g, return_inverse=True)
        assert np.all((g >= 0) & (g < len(raw)))
        assert all(identities[int(ii)]['index'] == int(ii) and identities[int(ii)]['source'] == 'CLC' and identities[int(ii)]['role'] == fold for ii in im)
        assert all(proposal[int(gg)]['file_name'] == identities[int(ii)]['file_name'] for gg, ii in zip(g, im))
        assert np.all(old['query_valid_'+fold][rows])
        assert len(np.unique(np.stack([g, cid], axis=1), axis=0)) == len(rows)
        assert len(rows) == len(unique)*12 and np.all(old['class_count_'+fold][rows] == 12)
        columns = np.asarray([cid_lookup[int(c)] for c in cid])
        _, replay = material_columns(np.asarray(raw[unique]), old_lookup, support)
        assert np.array_equal(replay[inv, columns].view(np.uint32), old['X_'+fold][rows, 8:12].view(np.uint32))
        _, changed = material_columns(np.asarray(raw[unique]), aligned_lookup, support)
        new['X_'+fold][rows, 8:12] = changed[inv, columns]
        keep = [0, 1, 2, 3, 4, 5, 6, 7, 12]
        assert np.array_equal(new['X_'+fold][:, keep].view(np.uint32), old['X_'+fold][:, keep].view(np.uint32))
        other = old['source_index_'+fold] != 0
        assert np.array_equal(new['X_'+fold][other].view(np.uint32), old['X_'+fold][other].view(np.uint32))
        fold_proofs.append(dict(fold=fold, source_CLC_tuples=len(rows), unique_proposals=len(unique),
            old_M8_11_replay_F32_bitdiff=0, unchanged_G_and_availability_F32_bitdiff=0,
            PID_Hi_all13_F32_bitdiff=0, changed_scalar_counts={str(j): int(np.count_nonzero(
                new['X_'+fold][rows, j].view(np.uint32) != old['X_'+fold][rows, j].view(np.uint32))) for j in [8, 9, 10, 11]}))
    for key in old:
        if key not in ['X_train', 'X_val', 'metadata_json']:
            assert np.array_equal(new[key], old[key]), key
    updated = dict(metadata, protocol_sha256=pr['sha256'], source_N_protocol_sha256=P['sources']['N_protocol']['sha256'],
        source_N_training=old_ref, bank_alignment_only_source_CLC=True, changed_source_coordinates=[8, 9, 10, 11],
        actual_current_class_lookup=P['sources']['runtime_lookup_reference'], auxiliary_alignment_does_not_rebuild_native_F=True)
    new['metadata_json'] = np.asarray(json.dumps(updated))
    training_path = OUT/'training/multisource_train_val.npz'
    savez(training_path, **new)
    receipt_path = OUT/'training/data_receipt.json'
    save(receipt_path, dict(status='COMPLETE', completed_utc=now(), protocol=pr, training_npz=ref(training_path),
        source_N_training=old_ref, source_N_training_receipt=P['sources']['N_training_receipt'],
        feature_names=old['feature_names'].astype(str).tolist(), runtime_lookup_reference=P['sources']['runtime_lookup_reference'],
        fold_proofs=fold_proofs, all_teacher_identity_classcount_queryvalid_unchanged=True,
        modified_sources=['CLC'], modified_coordinates=[8, 9, 10, 11], new_GT_decoder_encoder_GPU_AP=0))
    case = 'CLC_tune'; original_manifest = P['cases'][case]['N_feature_manifest']
    nf = read(verify(original_manifest)); cf = read(verify(nf['source_C_feature_manifest']))
    xold = np.load(verify(nf['outputs']['candidate_features.npy']), allow_pickle=False)
    valid = np.load(verify(nf['outputs']['candidate_valid.npy']), allow_pickle=False)
    ci = np.load(verify(nf['outputs']['candidate_class_order_index.npy']), allow_pickle=False)
    raw_ct = np.load(verify(nf['outputs']['fresh_exact22.npy']), mmap_mode='r', allow_pickle=False)
    trace = loadz(verify(cf['outputs']['source_trace.npz']))
    with open(verify(P['sources']['CLC_image_roles']), newline='') as handle:
        tune_roles = {int(r['original_image_id']): r for r in csv.DictReader(handle) if r['role'] == 'tune'}
    id_map = read(verify(cf['identity']['assets']['image_id_map']))
    assert all(int(oid) in tune_roles and tune_roles[int(oid)]['file_name'] == proposal[int(g)]['file_name'] and int(id_map[str(int(oid))]) == int(iid) for g, oid, iid in zip(trace['global_cache_index'], trace['original_image_id'], trace['image_id']))
    assert np.all(np.load(verify(nf['outputs']['query_valid.npy']), allow_pickle=False))
    assert all(int(proposal[int(g)]['original_image_id']) == int(oid) for g, oid in zip(trace['global_cache_index'], trace['original_image_id']))
    assert np.array_equal(np.asarray([proposal[int(g)]['bbox'] for g in trace['global_cache_index']], np.float64), trace['bbox'].astype(np.float64))
    assert np.array_equal(np.asarray(raw[trace['global_cache_index']]).view(np.uint64), np.asarray(raw_ct).view(np.uint64))
    old_B, old_columns = material_columns(np.asarray(raw_ct), old_lookup, support)
    assert np.array_equal(old_B.view(np.uint32), np.load(verify(nf['outputs']['fresh_material_B.npy'])).view(np.uint32))
    B, columns = material_columns(np.asarray(raw_ct), aligned_lookup, support)
    xnew = xold.copy(); replay = np.zeros(xold[:, :, 8:12].shape, np.float32)
    for slot in [0, 1]:
        rows = np.flatnonzero(valid[:, slot])
        replay[rows, slot] = old_columns[rows, ci[rows, slot]]
        xnew[rows, slot, 8:12] = columns[rows, ci[rows, slot]]
    assert np.array_equal(replay.view(np.uint32), xold[:, :, 8:12].view(np.uint32))
    assert np.array_equal(xnew[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12]].view(np.uint32), xold[:, :, [0, 1, 2, 3, 4, 5, 6, 7, 12]].view(np.uint32))
    assert not np.any(xnew[~valid]) and np.array_equal(xnew[:, :, 12], xold[:, :, 12])
    folder = OUT/'features/CLC_tune'
    array(folder/'candidate_features.npy', xnew); array(folder/'fresh_material_B.npy', B)
    outputs = dict(nf['outputs']); outputs.update({'candidate_features.npy': ref(folder/'candidate_features.npy'), 'fresh_material_B.npy': ref(folder/'fresh_material_B.npy')})
    fm = dict(nf, protocol=pr, protocol_sha256=pr['sha256'], outputs=outputs,
        source_N_feature_manifest=original_manifest, runtime_lookup_reference=P['sources']['runtime_lookup_reference'],
        auxiliary_bank_aligned_to=P['sources']['CLC_target_transfer_bank'], changed_coordinates=[8, 9, 10, 11],
        actual_bank=P['sources']['CLC_target_transfer_bank'],
        source_N_original_bank=nf['actual_bank'], source_N_original_support_counts=nf['actual_material_support_counts'],
        cached_exact22_reprojection_no_new_RGB=True,
        native_F_pool_original_bank=cf['identity']['assets']['material_bank'], native_F_pool_rebuilt=False)
    new_bank = read(verify(P['sources']['CLC_target_transfer_bank']))
    by_name = {p['category_name']: int(p['sample_count']) for p in new_bank['prototypes'].values()}
    fm['actual_material_support_counts'] = [by_name[c['category_name']] for c in classes]
    assert all(n > 0 for n in fm['actual_material_support_counts'])
    save(folder/'feature_manifest.json', fm)
    save(folder/'status.json', dict(status='COMPLETE', protocol=pr, feature_manifest=ref(folder/'feature_manifest.json')))
    # Prove exact effective inputs for all three old availability-control seeds.
    import torch
    controls = []
    for seed in P['seeds']:
        checkpoint_ref = P['sources']['T_checkpoints']['availability_control'][str(seed)]
        checkpoint = torch.load(verify(checkpoint_ref), map_location='cpu', weights_only=True)
        config = checkpoint['config']; assert config['protocol'] == P['sources']['T_protocol'] and config['variant'] == 'availability_control'
        proof = dict(seed=seed, checkpoint=checkpoint_ref, train_val={}, inference={})
        for fold in ['train', 'val']:
            for kind, norm in [('residual', config['residual_normalization']), ('anchor', config['anchor_normalization'])]:
                a, b = masked(old['X_'+fold], norm), masked(new['X_'+fold], norm)
                assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
                proof['train_val'][fold+'_'+kind] = dict(F32_bitdiff=0, effective_input_SHA256=hashlib.sha256(a.tobytes()).hexdigest())
        for case in P['inference_cases']:
            old_fm = read(verify(P['cases'][case]['N_feature_manifest']))
            oldx = np.load(verify(old_fm['outputs']['candidate_features.npy']), mmap_mode='r', allow_pickle=False)
            newx = xnew if case == 'CLC_tune' else oldx
            for kind, norm in [('residual', config['residual_normalization']), ('anchor', config['anchor_normalization'])]:
                a, b = masked(np.asarray(oldx), norm), masked(np.asarray(newx), norm)
                assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
            proof['inference'][case] = dict(effective_anchor_and_residual_F32_bitdiff=0,
                old_N_feature_manifest=P['cases'][case]['N_feature_manifest'], new_feature_manifest=ref(folder/'feature_manifest.json') if case == 'CLC_tune' else P['cases'][case]['N_feature_manifest'])
        controls.append(proof)
    bridge_path = OUT/'verification/control_effective_input_bridge.json'
    save(bridge_path, dict(status='PASS', protocol=pr, completed_utc=now(), source_N_training=old_ref,
        aligned_training=ref(training_path), source_T_protocol=P['sources']['T_protocol'], controls=controls,
        teacher_keys_weights_classcount_queryvalid_exact=True, source_sampler_draw_distribution_unchanged=True,
        availability_control_checkpoint_references=3, availability_control_inference_references=21,
        legacy_material_six_test_inference_references=18, legacy_material_new_CT_requires_actual_GPU=3,
        old_material_test_features_unchanged=True, new_GT_decoder_encoder_GPU_AP=0,
        caveat='Only auxiliary feature bank aligned; original CT native F candidates retain their old bank/UNC states.'))
    save(OUT/'data_completion.json', dict(status='COMPLETE_BANK_ALIGNED_AUXILIARY_DATA_READY', protocol=pr,
        completed_utc=now(), training=ref(receipt_path), CLC_tune_feature_manifest=ref(folder/'feature_manifest.json'),
        control_effective_input_bridge=ref(bridge_path), runtime_lookup_reference=P['sources']['runtime_lookup_reference'],
        only_source_CLC_and_CT_M8_11_changed=True, six_test_feature_arrays_original_references=True,
        new_GT_decoder_encoder_GPU_AP=0, publication_edits=False))
    return read(OUT/'data_completion.json')
