"""Frozen geometry anchor with a bounded, availability-gated residual.

This module writes only the new experiment root. It never reads a test label,
changes the detector, computes AP, or chooses a deployment residual strength.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
CODE = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_bank_alignment_20261101')
VARIANTS = ('material',)
ROLES = ('availability_control', 'material', 'legacy_material')
SEEDS = (17, 29, 43)
FEATURE_NAMES = [
    'confidence_percentile', 'clip_log_aspect_over4', 'width_over_image_width',
    'height_over_image_height', 'centerx_over_width', 'centery_over_height',
    'max_other_proposal_IoU', 'sqrt_relative_box_area', 'fresh_B_class',
    'supported_B_rank_over_T_minus1', 'supported_B_top_minus_second_over2',
    'B_class_minus_supported_top_over2', 'has_actual_material_support',
]
FORMAT = 'bank_aligned_availability_gated_frozen_geometry_residual_v1'
LEGACY_FORMAT = 'availability_gated_frozen_geometry_residual_v1'


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def verify(value):
    expected = {k: value[k] for k in ('path', 'bytes', 'sha256')}
    assert ref(value['path']) == expected, value['path']
    return Path(value['path'])


def writable(path, child):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT / child), (str(p), child)
    assert not (p.exists() and any(p.iterdir())), f'Refuse overwrite: {p}'
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic(path, value):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT), str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    os.replace(temp, p)


def save_npz(path, **arrays):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT), str(p)
    with p.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def save_checkpoint(path, value):
    p = Path(path).resolve()
    assert p.is_relative_to(OUT), str(p)
    temp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with temp.open('xb') as stream:
        torch.save(value, stream)
    os.replace(temp, p)


def protocol(path, expected_sha256):
    p = Path(path).resolve()
    assert p == CODE / 'config/protocol.json'
    pr = ref(p)
    assert len(expected_sha256) == 64 and pr['sha256'] == expected_sha256
    value = read(p)
    assert value['status'].startswith('FROZEN')
    t = value['training']
    assert t['variants'] == list(VARIANTS) and t['seeds'] == list(SEEDS)
    assert value['variants'] == list(ROLES)
    assert (t['learning_rate'], t['weight_decay'], t['max_epochs'],
            t['patience'], t['min_delta']) == (.001, .0001, 100, 15, 1e-6)
    assert (t['residual_bound'], t['residual_l2_penalty'], t['gate_final_bias']) == (2., .01, -2.)
    assert t['residual_zero_initialization'] is True and t['anchor_candidate_epoch'] == -1
    assert value['feature_schema']['names'] == FEATURE_NAMES
    assert value['model_design'] == ref(CODE / 'config/model_design.json')
    return value, pr


class GeometryAnchor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GatedResidual(nn.Module):
    def __init__(self, anchor_state):
        super().__init__()
        self.anchor = GeometryAnchor()
        self.anchor.load_state_dict(anchor_state)
        self.anchor.requires_grad_(False)
        self.gate = nn.Sequential(nn.Linear(13, 8), nn.ReLU(), nn.Linear(8, 1))
        self.residual = nn.Sequential(nn.Linear(13, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.)
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def components(self, anchor_x, residual_x, availability):
        anchor = self.anchor(anchor_x)
        gate = torch.sigmoid(self.gate(residual_x).squeeze(-1))
        bounded = 2. * torch.tanh(self.residual(residual_x).squeeze(-1))
        correction = availability * gate * bounded
        # The explicit where keeps absent-support logit bits identical to anchor.
        logits = torch.where(availability > 0, anchor + correction, anchor)
        return dict(anchor_logits=anchor, gate_weights=gate,
                    bounded_residual=bounded, correction_logits=correction,
                    availability=availability, full_logits=logits)

    def forward(self, anchor_x, residual_x, availability):
        values = self.components(anchor_x, residual_x, availability)
        return values['full_logits'], values['correction_logits']


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def masked(raw, active, constant=()):
    x = np.array(raw, dtype=np.float32, copy=True)
    assert x.ndim == 2 and x.shape[1] == 13 and np.isfinite(x).all()
    x[:, [j for j in range(13) if j not in active]] = 0.
    if constant:
        x[:, list(constant)] = 0.
    return x


def variant_active(variant):
    assert variant in VARIANTS
    if variant == 'material':
        return list(range(13))
    return list(range(8)) + ([9, 12] if variant == 'ordinal_material' else [12])


def availability(raw, query_valid=None):
    x = np.asarray(raw)
    assert np.all((x[:, 12] == 0.) | (x[:, 12] == 1.))
    a = x[:, 12].astype(np.float32, copy=True)
    if query_valid is not None:
        assert query_valid.dtype == bool and query_valid.shape == a.shape
        assert not np.any(a[~query_valid])
    return a


def populations(d):
    s, img, g, c, cid = [d[k] for k in ('source_index', 'image_index',
                                      'global_cache_index', 'class_count', 'category_id')]
    assert set(np.unique(s)) == {0, 1, 2} and np.all(c >= 2)
    keys = np.stack([s, img, g], 1)
    pk, pi, pc = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    assert np.array_equal(pc[pi], c)
    assert len(np.unique(np.stack([s, img, g, cid], 1), axis=0)) == len(s)
    ik, ii, ic = np.unique(np.stack([s, img], 1), axis=0,
                           return_inverse=True, return_counts=True)
    assert not np.any(ic[ii] % c)
    nprops = ic[ii] // c
    w = 1. / (nprops.astype(np.float64) * c.astype(np.float64))
    valw = np.empty(len(s), np.float64)
    groups, statistics = [], {}
    for sid in range(3):
        rows = np.flatnonzero(s == sid)
        classes = np.unique(c[rows]); assert len(classes) == 1
        images = np.unique(img[rows])
        localw = w[rows].copy()
        groups.append((rows, torch.from_numpy(localw)))
        valw[rows] = 1. / (3. * len(images) * ic[ii[rows]])
        statistics[str(sid)] = dict(valid_tuples=len(rows), effective_images=len(images),
                                   proposals=int(np.sum(pk[:, 0] == sid)), classes=int(classes[0]),
                                   sampler_weights_f64_sha256=hashlib.sha256(localw.tobytes()).hexdigest())
    assert abs(valw.sum() - 1.) < 1e-12
    return groups, valw, statistics


def dataset(P, pr):
    inp = P['training']['source_input']
    receipt_path = Path(inp['actual_ref_from']).resolve()
    assert receipt_path == OUT/'training/data_receipt.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'COMPLETE' and receipt['protocol'] == pr
    assert receipt['source_N_training'] == P['sources']['N_training']
    source_ref = receipt['training_npz']
    assert Path(inp['path']).resolve() == verify(source_ref) == OUT/'training/multisource_train_val.npz'
    with np.load(source_ref['path'], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert arrays['feature_names'].astype(str).tolist() == FEATURE_NAMES
    metadata = json.loads(str(arrays['metadata_json'].item()))
    assert metadata['protocol_sha256'] == pr['sha256']
    assert metadata['source_N_protocol_sha256'] == P['sources']['N_protocol']['sha256']
    data = {}
    keys = ('X', 'y', 'source_index', 'image_index', 'class_count', 'category_id',
            'global_cache_index', 'query_valid')
    for role in ('train', 'val'):
        d = {key: arrays[f'{key}_{role}'] for key in keys}
        count = len(d['y'])
        assert d['X'].dtype == np.float32 and d['X'].shape == (count, 13)
        assert d['y'].dtype == np.float32 and d['y'].shape == (count,)
        assert np.isfinite(d['X']).all() and np.isfinite(d['y']).all()
        assert np.all((d['y'] >= 0) & (d['y'] <= 1))
        for key in keys[2:-1]:
            assert d[key].dtype == np.int64 and d[key].shape == (count,)
        assert d['query_valid'].dtype == bool and d['query_valid'].shape == (count,)
        assert not np.any(d['X'][~d['query_valid'], 8:])
        groups, weights, stats = populations(d)
        d.update(groups=groups, balance_weights=weights, statistics=stats,
                 availability=availability(d['X'], d['query_valid']))
        data[role] = d
    assert not (set(zip(data['train']['source_index'], data['train']['image_index'])) &
                set(zip(data['val']['source_index'], data['val']['image_index'])))
    return data, metadata, receipt


def cuda(device, P):
    expected = P['resources']['GPU_UUID']
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected
    assert P['resources']['GPU_index'] in (0, 1), 'Only an explicitly reserved free GPU0/1'
    dev = torch.device(device)
    assert dev.type == 'cuda' and torch.cuda.is_available()
    torch.cuda.set_device(dev); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return dev


def load_anchor(P, seed, map_location='cpu'):
    anchor_ref = P['sources']['K_geometry_checkpoints'][str(seed)]
    checkpoint = torch.load(verify(anchor_ref), map_location=map_location, weights_only=True)
    config = checkpoint['config']
    assert (config['variant'], config['seed']) == ('geometry', seed)
    assert config['architecture'] == [13, 32, 16, 1]
    norm = config['normalization']
    assert norm['active_coordinates'] == list(range(8))
    assert set(norm['constant_coordinates']) >= set(range(8, 13))
    return checkpoint['state_dict'], norm, anchor_ref


def prepared_inputs(raw, variant, residual_constants, anchor_normalization, dev):
    anchor_x = masked(raw, anchor_normalization['active_coordinates'],
                      anchor_normalization['constant_coordinates'])
    residual_x = masked(raw, variant_active(variant), residual_constants)
    a = availability(raw)
    return (torch.from_numpy(anchor_x).to(dev), torch.from_numpy(residual_x).to(dev),
            torch.from_numpy(a).to(dev))


def components_cpu(model, tensors, batch=8192):
    model.eval()
    output = {k: [] for k in ('anchor_logits', 'gate_weights', 'bounded_residual',
                              'correction_logits', 'availability', 'full_logits')}
    with torch.no_grad():
        for lo in range(0, len(tensors[0]), batch):
            values = model.components(*(x[lo:lo + batch] for x in tensors))
            for key, value in values.items():
                output[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in output.items()}


def monitor(model, tensors, data):
    values = components_cpu(model, tensors)
    assert all(torch.isfinite(value).all() for value in values.values())
    logits, target = values['full_logits'], torch.from_numpy(data['y'])
    weighted = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.from_numpy((data['class_count'] - 1).astype(np.float32)),
        reduction='none').numpy().astype(np.float64)
    plain = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none').numpy().astype(np.float64)
    q = torch.sigmoid(logits).numpy().astype(np.float32)
    weights = data['balance_weights']
    delta = values['correction_logits'].numpy().astype(np.float64)
    metrics = dict(n=len(q), balanced_weighted_BCE=float(np.dot(weights, weighted)),
                   balanced_unweighted_BCE=float(np.dot(weights, plain)),
                   balanced_MSE=float(np.dot(weights, (q.astype(np.float64) - data['y']) ** 2)),
                   prediction_mean=float(np.dot(weights, q)), target_mean=float(np.dot(weights, data['y'])),
                   balanced_squared_correction=float(np.dot(weights, delta ** 2)),
                   max_abs_correction=float(np.abs(delta).max()),
                   unsupported_exact_anchor_logits=bool(torch.equal(
                       values['full_logits'][values['availability'] == 0],
                       values['anchor_logits'][values['availability'] == 0])), per_source={})
    assert metrics['unsupported_exact_anchor_logits']
    for source in range(3):
        rows = data['source_index'] == source
        localw = weights[rows] / weights[rows].sum()
        metrics['per_source'][str(source)] = dict(
            n=int(rows.sum()), weighted_BCE=float(np.dot(localw, weighted[rows])),
            unweighted_BCE=float(np.dot(localw, plain[rows])),
            MSE=float(np.dot(localw, (q[rows].astype(np.float64) - data['y'][rows]) ** 2)),
            squared_correction=float(np.dot(localw, delta[rows] ** 2)),
            availability_fraction=float(np.dot(localw, data['availability'][rows])))
    arrays = {k: v.numpy().astype(np.float32) for k, v in values.items()}
    arrays['quality'] = q
    return arrays, metrics


def data_gate(pr):
    gate=read(OUT/'verification/data_review.json')
    assert gate['status']=='PASS'and gate['protocol_sha256']==pr['sha256']


def train(a):
    P, pr = protocol(a.protocol, a.protocol_sha256)
    data_gate(pr)
    variant, seed = a.variant, a.seed
    assert variant in VARIANTS and seed in SEEDS
    out = writable(a.output_dir, 'models')
    assert out == OUT / 'models' / variant / f'seed{seed}'
    data, metadata, source_receipt = dataset(P, pr)
    dev = cuda(a.device, P)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    anchor_state, anchor_norm, anchor_ref = load_anchor(P, seed)
    anchor_digest = state_digest(anchor_state)
    active = variant_active(variant)
    constants = np.flatnonzero(np.ptp(masked(data['train']['X'], active), axis=0) == 0).tolist()
    normalization = dict(kind='natural13_no_fitted_scaling', active_coordinates=active,
                         constant_coordinates=constants, fit='Unique source training tuples only',
                         structural_availability_from_raw_X12_before_masking=True)
    tensors = {role: prepared_inputs(d['X'], variant, constants, anchor_norm, dev)
               for role, d in data.items()}
    target = torch.from_numpy(data['train']['y']).to(dev)
    pos_weight = torch.from_numpy((data['train']['class_count'] - 1).astype(np.float32)).to(dev)
    model = GatedResidual(anchor_state).to(dev)
    counts = dict(total=sum(p.numel() for p in model.parameters()),
                  trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  frozen_anchor=sum(p.numel() for p in model.anchor.parameters()))
    assert counts == dict(total=1355, trainable=362, frozen_anchor=993)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=.001, weight_decay=.0001)
    config = dict(protocol=pr, model_design=ref(CODE / 'config/model_design.json'),
                  input=source_receipt['training_npz'], data_receipt=ref(P['training']['source_input']['actual_ref_from']),
                  source_N_training=P['sources']['N_training'],
                  variant=variant, seed=seed, feature_names=FEATURE_NAMES,
                  anchor_checkpoint=anchor_ref, anchor_normalization=anchor_norm,
                  residual_normalization=normalization, parameters=counts,
                  residual_architecture=[13, 16, 1], gate_architecture=[13, 8, 1],
                  residual_bound=2., gate_last_bias_initial=-2., residual_last_layer_zero_initial=True,
                  residual_penalty=.01, training=P['training'], teacher_metadata=metadata,
                  source_statistics={role: d['statistics'] for role, d in data.items()},
                  view_count=2, draws_per_epoch=29370, updates_per_epoch=29,
                  epoch_minus1_anchor_is_checkpoint_candidate=True,
                  validation_selection='Equal source/image/tuple weighted BCE only; no residual penalty',
                  sigmoid_location='CPUFloat32 from actual GPU Float32 logits')
    environment = dict(PID=os.getpid(), PPID=os.getppid(), argv=sys.argv, started_utc=now(),
                       CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], GPU_name=torch.cuda.get_device_name(dev),
                       device=str(dev), torch_version=str(torch.__version__), deterministic_algorithms=True,
                       TF32_enabled=False)
    atomic(out / 'run_config.json', dict(config=config, environment=environment))
    atomic(out / 'normalization.json', normalization)
    atomic(out / 'run_status.json', dict(status='RUNNING_ACTUAL_CUDA', variant=variant, seed=seed, PID=os.getpid()))
    torch.cuda.reset_peak_memory_stats(dev); started = time.perf_counter()
    initial_arrays, initial_metrics = monitor(model, tensors['val'], data['val'])
    assert not np.any(initial_arrays['correction_logits'])
    assert np.array_equal(initial_arrays['full_logits'], initial_arrays['anchor_logits'])
    atomic(out / 'initial_metrics.json', dict(epoch=-1, val=initial_metrics, anchor_state_sha256=anchor_digest))
    save_npz(out / 'initial_val_natural_quality.npz', **initial_arrays)
    best, best_epoch, stale = initial_metrics['balanced_weighted_BCE'], -1, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_checkpoint(out / 'model.pt', dict(format=FORMAT, state_dict=best_state, config=config,
                                          best_epoch=best_epoch, best_val_BCE=best))
    history = []
    for epoch in range(100):
        if stale >= 15:
            break
        chunks = [torch.from_numpy(rows)[torch.multinomial(weights, 9790, replacement=True, generator=generator)]
                  for rows, weights in data['train']['groups']]
        draw = torch.cat(chunks); draw = draw[torch.randperm(29370, generator=generator)]
        assert len(draw) == 29370
        model.train(); model.anchor.eval(); steps = 0; objective_sum = 0.
        for indices in draw.split(1024):
            ii = indices.to(dev)
            xx = tuple(torch.cat([x[ii], x[ii]]) for x in tensors['train'])
            yy, ww = torch.cat([target[ii], target[ii]]), torch.cat([pos_weight[ii], pos_weight[ii]])
            optimizer.zero_grad(set_to_none=True)
            logits, correction = model(*xx)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yy, pos_weight=ww)
            loss = loss + .01 * torch.mean(correction.square())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite material reliability training objective')
            loss.backward(); optimizer.step(); objective_sum += float(loss.detach().cpu()); steps += 1
        assert steps == 29
        _, metrics = monitor(model, tensors['val'], data['val'])
        val_loss = metrics['balanced_weighted_BCE']
        if val_loss < best - 1e-6:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(out / 'model.pt', dict(format=FORMAT, state_dict=best_state, config=config,
                                                  best_epoch=best_epoch, best_val_BCE=best))
        else:
            stale += 1
        anchor_unchanged = state_digest(model.anchor.state_dict()) == anchor_digest
        assert anchor_unchanged
        counts_drawn = {str(s): int((data['train']['source_index'][draw.numpy()] == s).sum()) for s in range(3)}
        assert set(counts_drawn.values()) == {9790}
        record = dict(epoch=epoch, val_BCE=val_loss, val_metrics=metrics, best_epoch=best_epoch,
                      stale_epochs=stale, mean_batch_training_objective=objective_sum / steps,
                      optimizer_steps=steps, draws_per_source=counts_drawn,
                      draw_sha256=hashlib.sha256(draw.numpy().tobytes()).hexdigest(),
                      anchor_weights_unchanged=anchor_unchanged, elapsed_seconds=time.perf_counter() - started)
        history.append(record); atomic(out / 'train_history.json', history)
        save_checkpoint(out / 'last.pt', dict(format=FORMAT, config=config, epoch=epoch,
                                             state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                             optimizer_state=optimizer.state_dict(), best_state=best_state,
                                             best_loss=best, best_epoch=best_epoch, stale=stale, history=history,
                                             draw_generator_state=generator.get_state()))
        print(json.dumps(record), flush=True)
    model.load_state_dict(best_state)
    assert state_digest(model.anchor.state_dict()) == anchor_digest
    metrics = {}
    for role in ('train', 'val'):
        arrays, metrics[role] = monitor(model, tensors[role], data[role])
        save_npz(out / f'{role}_natural_quality.npz', **arrays)
    assert metrics['val']['balanced_weighted_BCE'] <= initial_metrics['balanced_weighted_BCE'] + 1e-12
    torch.cuda.synchronize(dev)
    result = dict(status='COMPLETE', completed_utc=now(), variant=variant, seed=seed, actual_CUDA=True,
                  environment=environment, checkpoint=ref(out / 'model.pt'), best_epoch=best_epoch,
                  best_val_BCE=best, initial_val_BCE=initial_metrics['balanced_weighted_BCE'],
                  initial_anchor_retained=(best_epoch == -1), epochs_completed=len(history),
                  optimizer_updates=29 * len(history), parameters=counts,
                  anchor_state_sha256_before=anchor_digest,
                  anchor_state_sha256_after=state_digest(model.anchor.state_dict()),
                  anchor_weights_unchanged=True, metrics=metrics,
                  elapsed_seconds=time.perf_counter() - started,
                  peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(dev),
                  test_labels_read=False, AP_computed=False, deployment_parameters_selected=False)
    atomic(out / 'metrics.json', result)
    atomic(out / 'run_status.json', {key: result[key] for key in (
        'status', 'variant', 'seed', 'actual_CUDA', 'best_epoch', 'epochs_completed',
        'optimizer_updates', 'anchor_weights_unchanged', 'test_labels_read', 'AP_computed')})
    return result


def inference_feature(case, P, pr):
    assert case in P['cases']
    fp=Path(P['cases'][case]['feature_manifest_path']).resolve()
    if case=='CLC_tune':
        assert fp==OUT/'features/CLC_tune/feature_manifest.json'
        state=read(OUT/'features/CLC_tune/status.json')
        assert state['status']=='COMPLETE'and state['protocol']==pr and state['feature_manifest']==ref(fp)
        F=read(fp);assert F['protocol']==pr
    else:
        assert ref(fp)==P['cases'][case]['N_feature_manifest']
        F=read(fp);assert F['protocol']==P['sources']['N_protocol']
    assert F['case']==case and F['feature_names']==FEATURE_NAMES and F['dtype']=='float32'
    assert F['candidate_row_key_sha256']==P['cases'][case]['candidate_row_key_sha256']
    return fp,F


def infer(a):
    P, pr = protocol(a.protocol, a.protocol_sha256)
    data_gate(pr)
    role=a.role
    assert role in ('material','legacy_material')
    assert role!='legacy_material' or a.case=='CLC_tune', 'Legacy test inference is a strict old reference, never a new forward'
    fp,F=inference_feature(a.case,P,pr)
    x_path=verify(F['outputs']['candidate_features.npy'])
    valid_path=verify(F['outputs']['candidate_valid.npy'])
    X=np.load(x_path,mmap_mode='r',allow_pickle=False)
    M=np.load(valid_path,allow_pickle=False)
    assert X.dtype==np.float32 and X.shape[1:]==(2,13)
    assert M.dtype==bool and M.shape==X.shape[:2] and list(X.shape)==F['shape']
    checkpoint_path=Path(a.checkpoint).resolve()
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    config=checkpoint['config'];seed=config['seed'];assert seed in SEEDS and config['variant']=='material'
    if role=='material':
        assert checkpoint_path==OUT/'models/material'/f'seed{seed}'/'model.pt'
        assert checkpoint['format']==FORMAT and config['protocol']==pr
    else:
        assert ref(checkpoint_path)==P['sources']['T_checkpoints']['material'][str(seed)]
        assert checkpoint['format']==LEGACY_FORMAT and config['protocol']==P['sources']['T_protocol']
    assert config['feature_names']==FEATURE_NAMES
    anchor_state,anchor_norm,anchor_ref=load_anchor(P,seed)
    assert config['anchor_checkpoint']==anchor_ref and config['anchor_normalization']==anchor_norm
    dev=cuda(a.device,P)
    model=GatedResidual(anchor_state);model.load_state_dict(checkpoint['state_dict']);model.to(dev);model.eval()
    assert state_digest(model.anchor.state_dict())==state_digest(anchor_state)
    out=writable(a.output_dir,'inference')
    assert out==OUT/'inference'/a.case/role/f'seed{seed}'
    output={key:np.zeros(M.shape,np.float32)for key in (
        'quality','anchor_logits','gate_weights','bounded_residual','correction_logits','availability','full_logits')}
    started=time.perf_counter()
    for lo in range(0,len(X),8192):
        hi=min(lo+8192,len(X));local_mask=M[lo:hi];raw=np.asarray(X[lo:hi][local_mask])
        if not len(raw):continue
        tensors=prepared_inputs(raw,'material',config['residual_normalization']['constant_coordinates'],anchor_norm,dev)
        values=components_cpu(model,tensors)
        q=torch.sigmoid(values['full_logits']).numpy().astype(np.float32)
        for key,value in values.items():output[key][lo:hi][local_mask]=value.numpy().astype(np.float32)
        output['quality'][lo:hi][local_mask]=q
    for key,value in output.items():
        assert value.dtype==np.float32 and np.isfinite(value).all() and not np.any(value[~M])
    absent=M&(output['availability']==0)
    assert np.array_equal(output['full_logits'][absent],output['anchor_logits'][absent])
    assert not np.any(output['correction_logits'][absent])
    assert np.all((output['quality']>=0)&(output['quality']<=1))
    qp=out/'quality_scores.npz';save_npz(qp,**output)
    torch.cuda.synchronize(dev)
    result=dict(status='COMPLETE_LABEL_FREE_SHARED_UNION_INFERENCE',completed_utc=now(),
        PID=os.getpid(),argv=sys.argv,protocol=pr,protocol_sha256=pr['sha256'],
        case=a.case,variant=role,role=role,native_variant='material',seed=seed,
        native_model_protocol=config['protocol'],feature_manifest=ref(fp),
        candidate_row_key_sha256=F['candidate_row_key_sha256'],checkpoint=ref(checkpoint_path),
        anchor_checkpoint=anchor_ref,output=ref(qp),output_sha256=ref(qp)['sha256'],
        shape=list(M.shape),dtype='float32',invalid_quality_exact0=True,
        unsupported_exact_anchor_logits=True,new_head_components_saved=True,
        sigmoid_location='CPUFloat32 from actual GPU Float32 logits',actual_CUDA=True,
        new_GPU_inference=True,reference_reused=False,actual_fresh_GPU=True,strict_reuse=False,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        device=str(dev),GPU_name=torch.cuda.get_device_name(dev),torch_version=str(torch.__version__),CPU_threads=torch.get_num_threads(),elapsed_seconds=time.perf_counter()-started,
        GT_opened=False,labels_read=False,AP_computed=False,new_training=0,deployment_parameters_selected=False,
        source_T_checkpoint_bridge=(role=='legacy_material'),new_checkpoint_training=(role=='material'))
    atomic(out/'inference.json',result)
    return result
