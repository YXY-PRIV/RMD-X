"""Versioned inference continuation: fresh anchor output is diagnostic only.

Preparation is label-free outside the source train/val population. This module
has no evaluator, test-annotation reader, literature claims or paper writes.
Actual numerical entry points require the frozen protocol and independent gates.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
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
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_conditional_ranking_fix1_20261101')
VARIANTS = ('material', 'permuted_material')
SEEDS = (17, 29, 43)
FORMAT = 'conditional_supported_class_ranker_v1'
PROPOSAL_CHUNK = 8192
GPU_TUPLES = 8192
SOURCE_NAMES = ('CLC', 'PID', 'Hi')


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def ref(path):
    p = Path(path).resolve(); h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''): h.update(block)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())
def verify(value):
    assert ref(value['path']) == {k: value[k] for k in ('path', 'bytes', 'sha256')}, value['path']
    return Path(value['path'])
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bits(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
        np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(b).view(np.uint8))
def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def atomic(path, value):
    p = Path(path).resolve(); assert p.is_relative_to(OUT)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with q.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False); stream.write('\n')
    os.replace(q, p)
def save_npz(path, **values):
    p = Path(path).resolve(); assert p.is_relative_to(OUT)
    with p.open('xb') as stream: np.savez_compressed(stream, **values)
def save_array(path, value):
    p = Path(path).resolve(); assert p.is_relative_to(OUT)
    with p.open('xb') as stream: np.save(stream, value, allow_pickle=False)
def checkpoint(path, value):
    p = Path(path).resolve(); assert p.is_relative_to(OUT)
    q = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with q.open('xb') as stream: torch.save(value, stream)
    os.replace(q, p)
def fresh(path, child):
    p = Path(path).resolve(); assert p.is_relative_to(OUT/child)
    assert not p.exists() or not any(p.iterdir()), f'No implicit numerical retry: {p}'
    p.mkdir(parents=True, exist_ok=True); return p

def protocol(path, expected_sha256):
    p = Path(path).resolve(); assert p == CODE/'config/protocol.json'
    pr = ref(p); assert len(expected_sha256) == 64 and pr['sha256'] == expected_sha256
    P = read(p); assert P['status'].startswith('FROZEN')
    assert P['variants'] == list(VARIANTS) and P['seeds'] == list(SEEDS)
    t = P['training']; assert t['variants'] == list(VARIANTS) and t['seeds'] == list(SEEDS)
    assert (t['learning_rate'], t['weight_decay'], t['max_epochs'], t['patience'], t['min_delta']) == (.001, .0001, 100, 15, 1e-6)
    assert (t['residual_bound'], t['residual_l2_penalty'], t['gate_final_bias']) == (2., .01, -2.)
    assert (t['positive_proposal_draws_per_epoch'], t['regularizer_proposal_draws_per_epoch'], t['draws_per_source_per_population'], t['batch_size'], t['updates_per_epoch']) == (2937, 2937, 979, 256, 12)
    assert t['anchor_candidate_epoch'] == -1 and t['residual_zero_initialization'] is True
    assert P['model_design'] == ref(CODE/'config/model_design.json')
    assert P['budget']['new_trainings'] == 6 and P['budget']['new_full_C_GPU_inferences'] == 42
    for key in ['T_model_source', 'T_protocol', 'N_supported_source']: verify(P['sources'][key])
    assert P['resources']['GPU_index'] == 0 and P['resources']['CPU_threads'] == 8
    return P, pr

def cuda(device, P):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == P['resources']['GPU_UUID']
    dev = torch.device(device); assert dev.type == 'cuda' and torch.cuda.is_available()
    torch.cuda.set_device(dev); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True); return dev


def actor_gate(pr):
    """The directly callable numerical entry points enforce the controller lock-in."""
    freeze_path = OUT/'automation/model_code_freeze.json'
    freeze = read(freeze_path); assert freeze['protocol'] == pr
    required = {str(CODE/p) for p in ['src/quality_model.py', 'scripts/train_models.py',
        'scripts/run_models.py', 'config/model_design.json', 'src/group_data.py', 'config/protocol.json']}
    assert required <= {r['path'] for r in freeze['sources']}
    for key in ['sources', 'readonly_sources']:
        for r in freeze[key]: verify(r)
    for r in freeze['independent_reviews']:
        review = read(verify(r)); assert review['status'] == 'PASS' and review['protocol_sha256'] == pr['sha256']
    verify(freeze['training_npz']); verify(freeze['training_receipt'])
    return freeze


def continuation_gate(pr):
    path = OUT/'continuation/anchor_diagnostic_guard/verification_amendment.json'
    amendment_ref = ref(path)
    expected = os.environ.get('RAXO_CONTINUATION_AMENDMENT_SHA256', '')
    assert len(expected) == 64 and amendment_ref['sha256'] == expected
    amendment = read(path)
    assert amendment['status'] == 'FROZEN_VERIFICATION_ONLY_AMENDMENT_BEFORE_ANY_AP'
    assert amendment['study_protocol'] == pr and amendment['amendment_id'] == 'fresh_anchor_diagnostic_only_v1'
    assert amendment['training_data_parameters_math_unchanged'] is True
    assert amendment['original_limit'] == 1e-6 and amendment['observed_diff'] == 1.430511474609375e-6
    assert amendment['guard_policy'] == dict(
        unused_fresh_anchor='finite_and_record_max_abs_and_ULP_not_deployment_acceptance',
        deployed_candidate_anchor='bitexact_original_saved_Tfull_logits',
        frozen_T1355_state='exact_checkpoint_state_digest')
    assert (amendment['remaining_jobs'], amendment['completed_prior_jobs'],
            amendment['actual_total_inference_attempts'], amendment['scientific_completed_inferences']) == (24,18,43,42)
    required = {str(CODE/p) for p in ['src/quality_model_continued.py',
        'scripts/infer_continued.py', 'scripts/run_models_continued.py']}
    assert {r['path'] for r in amendment['sources']} == required
    for r in amendment['sources']: verify(r)
    for key in ['retained_inference_index', 'retained_training_complete', 'partial_archive_manifest']: verify(amendment[key])
    gate_path = OUT/'verification/continued_model_preflight_review.json'
    gate = read(gate_path)
    assert gate['status'] == 'PASS' and gate['protocol_sha256'] == pr['sha256']
    assert gate['verification_amendment'] == amendment_ref
    return amendment, amendment_ref, ref(gate_path)


def float32_ULP_distance(a, b):
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    def ordered(value):
        u = np.ascontiguousarray(value).view(np.uint32).astype(np.uint64)
        return np.where((u & np.uint64(0x80000000)) != 0,
            np.uint64(0xffffffff)-u, u+np.uint64(0x80000000)).astype(np.int64)
    return np.abs(ordered(a)-ordered(b))


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def load_anchor(P, seed):
    source = module(verify(P['sources']['T_model_source']), '_conditional_readonly_T_model')
    r = P['sources']['T_checkpoints']['availability_control'][str(seed)]
    ck = torch.load(verify(r), map_location='cpu', weights_only=True)
    c = ck['config']; assert ck['format'] == source.FORMAT
    assert c['protocol'] == P['sources']['T_protocol'] and (c['variant'], c['seed']) == ('availability_control', seed)
    state = {key[len('anchor.'):]: value for key, value in ck['state_dict'].items() if key.startswith('anchor.')}
    anchor = source.GatedResidual(state); anchor.load_state_dict(ck['state_dict']); anchor.requires_grad_(False); anchor.eval()
    assert sum(p.numel() for p in anchor.parameters()) == 1355
    assert c['residual_normalization']['active_coordinates'] == list(range(8)) + [12]
    return anchor, c, r


def normalized(x, constants, active=None):
    """Tensor constant guards fitted on original source-train values only."""
    out = x.clone()
    if active is not None: out[..., [j for j in range(13) if j not in active]] = 0.
    if constants: out[..., list(constants)] = 0.
    return out


class ConditionalRanker(nn.Module):
    def __init__(self, frozen_anchor):
        super().__init__()
        self.anchor = frozen_anchor; self.anchor.requires_grad_(False)
        self.gate = nn.Sequential(nn.Linear(13, 8), nn.ReLU(), nn.Linear(8, 1))
        self.residual = nn.Sequential(nn.Linear(13, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.constant_(self.gate[-1].bias, -2.)
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def old_logits(self, raw, anchor_config):
        shape = raw.shape[:-1]; flat = raw.reshape(-1, 13)
        g = anchor_config['anchor_normalization']; r = anchor_config['residual_normalization']
        gx = normalized(flat, g['constant_coordinates'], g['active_coordinates'])
        rx = normalized(flat, r['constant_coordinates'], r['active_coordinates'])
        with torch.no_grad(): values = self.anchor.components(gx, rx, flat[:, 12])
        return values['full_logits'].reshape(shape)

    def raw_components(self, residual_raw, context_raw, support, guards):
        b, c, _ = residual_raw.shape
        rx = normalized(residual_raw, guards['residual_constant_coordinates'])
        context = normalized(context_raw, guards['context_constant_coordinates'])
        gate = torch.sigmoid(self.gate(context).squeeze(-1))
        bounded = 2. * torch.tanh(self.residual(rx.reshape(-1, 13)).reshape(b, c))
        raw = support.to(torch.float32) * gate[:, None] * bounded
        return raw, gate, bounded

    def components(self, original_raw, residual_raw, context, support, guards, anchor_config):
        anchor = self.old_logits(original_raw, anchor_config)
        raw, gate, bounded = self.raw_components(residual_raw, context, support, guards)
        count = support.sum(1)
        # Full canonical catalogue tensor is used; unsupported raw terms are zero.
        mean = raw.to(torch.float64).sum(1) / count.clamp_min(1).to(torch.float64)
        centered = ((raw.to(torch.float64) - mean[:, None]) * .5).to(torch.float32)
        centered = torch.where(support & (count[:, None] > 1), centered, torch.zeros_like(centered))
        # A value-dependent zero test would kill all initial ranking gradients:
        # the new residual is deliberately initialized to exactly zero. Exact
        # deployment-quality fallback is applied later, outside this train graph.
        logits = anchor + centered
        return dict(anchor_logits=anchor, raw_correction=raw, supported_mean=mean,
                    centered_correction=centered, full_logits=logits, gate_weights=gate,
                    bounded_residual=bounded, support=support)


def pairwise_per_proposal(logits, y, support):
    """Multi-positive IoU-weighted positive-versus-zero softplus, no pos/pos pairs."""
    positive = support & (y > 0.); negative = support & (y == 0.)
    weights = y * positive.to(y.dtype)
    difference = logits[:, :, None] - logits[:, None, :]
    terms = nn.functional.softplus(-difference) * weights[:, :, None] * negative[:, None, :].to(logits.dtype)
    denominator = weights.sum(1) * negative.sum(1).to(logits.dtype)
    return terms.sum((1, 2)) / denominator.clamp_min(1e-30), denominator > 0


def regularizer_per_proposal(centered, support):
    return (centered.square() * support.to(centered.dtype)).sum(1) / support.sum(1).clamp_min(1).to(centered.dtype)


def permutation_indices(support, keys, seed, phase, epoch=None):
    """Label-independent per-proposal PCG64 permutations of supported four-tuples.

    SHA256 domain/seed/phase/epoch/proposal-key supplies a local 128-bit RNG seed.
    Fixed points are allowed. The identity mapping is retained outside support.
    """
    assert support.dtype == bool and keys.dtype == np.int64 and len(keys) == len(support)
    assert (phase == 'train') == (epoch is not None)
    n, c = support.shape; result = np.broadcast_to(np.arange(c, dtype=np.int64), (n, c)).copy()
    for i in range(n):
        active = np.flatnonzero(support[i])
        if len(active) < 2: continue
        payload = json.dumps(['conditional_material_permutation_v1', int(seed), str(phase),
            None if epoch is None else int(epoch), *[int(v) for v in keys[i]]], separators=(',', ':')).encode()
        entropy = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')
        result[i, active] = np.random.Generator(np.random.PCG64(entropy)).permutation(active)
    return result


def apply_permutation(x, indices):
    # Whole material tuple moves together; G, X12, labels and catalogue never move.
    out = x.clone(); ii = torch.as_tensor(indices, dtype=torch.long, device=x.device)
    out[:, :, 8:12] = torch.gather(x[:, :, 8:12], 1, ii[:, :, None].expand(-1, -1, 4))
    return out


def role_permutation(d, variant, seed, phase, epoch=None):
    if variant == 'material': return np.broadcast_to(np.arange(d['X'].shape[1], dtype=np.int64), d['support'].shape).copy()
    assert variant == 'permuted_material'
    keys = np.stack([d[k] for k in ['source_index', 'image_index', 'global_cache_index']], 1)
    return permutation_indices(d['support'], keys, seed, phase, epoch)


def dataset(P, pr):
    receipt_path = Path(P['training']['source_input']['actual_ref_from']); receipt = read(receipt_path)
    assert receipt['status'] == 'COMPLETE_GROUPED_SOURCE_DATA' and receipt['protocol'] == pr
    assert Path(P['training']['source_input']['path']) == verify(receipt['training_npz'])
    assert receipt['source_aligned_training'] == P['sources']['aligned_training']
    gate = read(OUT/'verification/data_review.json')
    assert gate['status'] == 'PASS' and gate['protocol_sha256'] == pr['sha256']
    source = module(CODE/'src/group_data.py', '_conditional_group_data')
    with np.load(receipt['training_npz']['path'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
    metadata = json.loads(str(arrays['metadata_json'].item())); assert metadata['protocol'] == pr
    keys = ['X', 'y', 'class_valid', 'support', 'source_index', 'image_index', 'global_cache_index',
            'class_count', 'category_id', 'class_order_index', 'query_valid', 'source_row_index',
            'eligible', 'conditional_weight', 'regularizer_weight', 'gate_context', 'supported_count']
    data = {}
    for fold in ['train', 'val']:
        d = {k: arrays[k+'_'+fold] for k in keys}; x, y, s = d['X'], d['y'], d['support']
        assert x.dtype == y.dtype == np.float32 and x.shape[1:] == (12, 13) and y.shape == x.shape[:2]
        assert np.isfinite(x).all() and np.isfinite(y).all() and s.dtype == bool and s.shape == y.shape
        assert np.array_equal(s, d['class_valid'] & (x[:, :, 12] == 1.) & d['query_valid'][:, None])
        eligible = ((y > 0.) & s).any(1) & ((y == 0.) & s).any(1) & (s.sum(1) >= 2)
        assert np.array_equal(eligible, d['eligible']) and not np.any(x[~d['class_valid']])
        context, count = source.gate_context(x, s, d['query_valid'])
        assert bits(context, d['gate_context']) and bits(count, d['supported_count'])
        for name, mask in [('conditional_weight', eligible), ('regularizer_weight', np.ones(len(x), bool))]:
            assert d[name].dtype == np.float64 and not np.any(d[name][~mask])
            for sid in range(3): assert abs(d[name][d['source_index'] == sid].sum() - 1./3.) < 1e-12
        data[fold] = d
    assert not (set(zip(data['train']['source_index'], data['train']['image_index'])) & set(zip(data['val']['source_index'], data['val']['image_index'])))
    return data, metadata, receipt, ref(receipt_path)


def sample_epoch(d, generator, population):
    weights = d[population]; chunks = []
    for sid in range(3):
        rows = np.flatnonzero((d['source_index'] == sid) & (weights > 0.))
        local = torch.from_numpy(weights[rows].copy())
        chosen = torch.multinomial(local, 979, replacement=True, generator=generator)
        chunks.append(torch.from_numpy(rows)[chosen])
    draws = torch.cat(chunks); draws = draws[torch.randperm(len(draws), generator=generator)]
    assert len(draws) == 2937
    assert all(int((d['source_index'][draws.numpy()] == sid).sum()) == 979 for sid in range(3))
    return draws


def monitor(model, d, tensors, permutation, guards, anchor_config):
    model.eval(); chunks = {k: [] for k in ['anchor_logits', 'raw_correction', 'supported_mean',
        'centered_correction', 'full_logits', 'gate_weights', 'bounded_residual']}
    with torch.no_grad():
        for lo in range(0, len(d['X']), 256):
            hi = min(lo+256, len(d['X'])); x, context, support = (v[lo:hi] for v in tensors)
            rx = apply_permutation(x, permutation[lo:hi])
            values = model.components(x, rx, context, support, guards, anchor_config)
            for k in chunks: chunks[k].append(values[k].cpu())
    arrays = {k: torch.cat(v).numpy() for k, v in chunks.items()}
    assert all(np.isfinite(v).all() for v in arrays.values())
    logits = torch.from_numpy(arrays['full_logits']); sy = torch.from_numpy(d['y']); ss = torch.from_numpy(d['support'])
    loss, eligible = pairwise_per_proposal(logits, sy, ss); assert np.array_equal(eligible.numpy(), d['eligible'])
    loss = loss.numpy().astype(np.float64)
    penalty = regularizer_per_proposal(torch.from_numpy(arrays['centered_correction']), ss).numpy().astype(np.float64)
    cw, rw = d['conditional_weight'], d['regularizer_weight']
    metrics = dict(conditional_pairwise_loss=float(np.dot(cw, loss)), full_population_squared_centered_delta=float(np.dot(rw, penalty)),
        selection_includes_regularizer=False, eligible_proposals=int(eligible.sum()), total_proposals=len(loss),
        padded_outputs_are_diagnostic_dummy_values_not_class_predictions=True,
        padding_excluded_from_all_losses_by_class_support_mask=True, per_source={})
    for sid, name in enumerate(SOURCE_NAMES):
        mask = d['source_index'] == sid
        metrics['per_source'][name] = dict(conditional_pairwise_loss=float(np.dot(cw[mask]*3., loss[mask])),
            full_population_squared_centered_delta=float(np.dot(rw[mask]*3., penalty[mask])), proposals=int(mask.sum()), eligible=int(d['eligible'][mask].sum()))
    absent = ~d['support'] | (d['supported_count'][:, None] <= 1)
    assert not np.any(arrays['centered_correction'][absent])
    assert bits(arrays['full_logits'][arrays['centered_correction'] == 0], arrays['anchor_logits'][arrays['centered_correction'] == 0])
    arrays.update(quality=torch.sigmoid(logits).numpy().astype(np.float32), support=d['support'], permutation_index=permutation)
    return arrays, metrics


def train(a):
    P, pr = protocol(a.protocol, a.protocol_sha256); seed, variant = a.seed, a.variant
    actor_gate(pr)
    assert seed in SEEDS and variant in VARIANTS
    out = fresh(a.output_dir, 'models'); assert out == OUT/'models'/variant/f'seed{seed}'
    data, metadata, source_receipt, receipt_ref = dataset(P, pr)
    dev = cuda(a.device, P); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Distinct draw domains, identical across the two roles; permutations use no draw RNG.
    cg = torch.Generator(device='cpu').manual_seed(seed)
    rg = torch.Generator(device='cpu').manual_seed(seed+1000003)
    anchor, anchor_config, anchor_ref = load_anchor(P, seed); anchor_digest = state_digest(anchor.state_dict())
    model = ConditionalRanker(anchor).to(dev)
    counts = dict(total=1717, trainable=362, frozen_anchor=1355)
    assert sum(p.numel() for p in model.parameters()) == 1717 and sum(p.numel() for p in model.parameters() if p.requires_grad) == 362
    unique = data['train']['X'][data['train']['class_valid']]
    guards = dict(kind='natural_units_constant_guard_only_from_original_source_train',
        residual_constant_coordinates=np.flatnonzero(np.ptp(unique, axis=0) == 0.).tolist(),
        context_constant_coordinates=np.flatnonzero(np.ptp(data['train']['gate_context'], axis=0) == 0.).tolist(),
        residual_fit_population='All original train valid catalogue tuples before any permutation',
        context_fit_population='All original train proposals before permutation', permutation_before_constant_guard=True)
    tensors = {fold: (torch.from_numpy(d['X']).to(dev), torch.from_numpy(d['gate_context']).to(dev), torch.from_numpy(d['support']).to(dev)) for fold, d in data.items()}
    yy = torch.from_numpy(data['train']['y']).to(dev)
    val_perm = role_permutation(data['val'], variant, seed, 'val')
    config = dict(protocol=pr, model_design=ref(CODE/'config/model_design.json'), variant=variant, seed=seed, parameters=counts,
        input=source_receipt['training_npz'], data_receipt=receipt_ref, teacher_metadata=metadata,
        anchor_checkpoint=anchor_ref, anchor_config=anchor_config, anchor_state_sha256=anchor_digest,
        guards=guards, training=P['training'], residual_architecture=[13,16,1], gate_architecture=[13,8,1],
        gate_kind='proposal_common_unpermuted_material_bag_context', residual_bound=2., center_divisor=2.,
        validation_selection='Pure full-source-val conditional pairwise loss; no regularizer; source/image/eligible-proposal equal',
        val_permutation_sha256=digest(val_perm), conditional_draw_RNG_seed=seed, regularizer_draw_RNG_seed=seed+1000003,
        permutation_algorithm='sha256_first128_little_endian_PCG64_v1', epoch_minus1_candidate=True)
    environment = dict(PID=os.getpid(), argv=sys.argv, started_utc=now(), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        GPU_name=torch.cuda.get_device_name(dev), torch_version=str(torch.__version__), device=str(dev), deterministic_algorithms=True, TF32_enabled=False)
    atomic(out/'run_config.json', dict(config=config, environment=environment)); atomic(out/'normalization.json', guards)
    atomic(out/'run_status.json', dict(status='RUNNING_ACTUAL_CONDITIONAL_CUDA_TRAINING', variant=variant, seed=seed, PID=os.getpid()))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.0001)
    torch.cuda.reset_peak_memory_stats(dev); start = time.perf_counter()
    initial_arrays, initial_metrics = monitor(model, data['val'], tensors['val'], val_perm, guards, anchor_config)
    assert not np.any(initial_arrays['raw_correction']) and not np.any(initial_arrays['centered_correction'])
    save_npz(out/'initial_val_components.npz', **initial_arrays); atomic(out/'initial_metrics.json', dict(epoch=-1, val=initial_metrics))
    best, best_epoch, stale = initial_metrics['conditional_pairwise_loss'], -1, 0
    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    def save_best(): checkpoint(out/'model.pt', dict(format=FORMAT, state_dict=best_state, config=config, best_epoch=best_epoch, best_val_conditional_loss=best))
    save_best(); history = []; first_update_evidence = None
    for epoch in range(100):
        if stale >= 15: break
        cd = sample_epoch(data['train'], cg, 'conditional_weight'); rd = sample_epoch(data['train'], rg, 'regularizer_weight')
        perm = role_permutation(data['train'], variant, seed, 'train', epoch)
        model.train(); model.anchor.eval(); objective = 0.; steps = 0
        for cp, rp in zip(cd.split(256), rd.split(256)):
            ci, ri = cp.to(dev), rp.to(dev); optimizer.zero_grad(set_to_none=True)
            cx, cc, cs = (v[ci] for v in tensors['train']); rx = apply_permutation(cx, perm[cp.numpy()])
            cv = model.components(cx, rx, cc, cs, guards, anchor_config)
            pair, eligible = pairwise_per_proposal(cv['full_logits'], yy[ci], cs); assert bool(eligible.all())
            gx, gc, gs = (v[ri] for v in tensors['train']); grx = apply_permutation(gx, perm[rp.numpy()])
            gv = model.components(gx, grx, gc, gs, guards, anchor_config)
            penalty = regularizer_per_proposal(gv['centered_correction'], gs)
            loss = pair.mean() + .01 * penalty.mean()
            assert bool(torch.isfinite(loss))
            first_update = (epoch == 0 and steps == 0)
            if first_update:
                last_weight_before = model.residual[-1].weight.detach().clone()
                assert not bool(torch.any(last_weight_before))
                assert not bool(torch.any(cv['centered_correction']))
                assert float(penalty.mean().detach().cpu()) == 0.
            loss.backward()
            if first_update:
                last_gradient = model.residual[-1].weight.grad
                assert last_gradient is not None and bool(torch.isfinite(last_gradient).all())
                gradient_norm = float(last_gradient.detach().to(torch.float64).norm().cpu())
                assert gradient_norm > 0., 'Zero-init conditional pairwise gradient is blocked'
            optimizer.step()
            if first_update:
                last_weight_after = model.residual[-1].weight.detach()
                update_norm = float((last_weight_after-last_weight_before).to(torch.float64).norm().cpu())
                assert update_norm > 0., 'First residual last-layer weight did not learn'
                assert state_digest(model.anchor.state_dict()) == anchor_digest
                first_update_evidence = dict(status='PASS_ACTUAL_INITIAL_PAIRWISE_GRADIENT_AND_UPDATE',
                    protocol=pr, variant=variant, seed=seed, actual_CUDA=True, epoch=epoch, optimizer_step=1,
                    conditional_draw_indices_SHA256=digest(cp.numpy()), regularizer_draw_indices_SHA256=digest(rp.numpy()),
                    initial_new_delta_exact0=True, initial_regularizer_exact0=True,
                    initial_residual_last_weight_exact0=True, residual_last_weight_gradient_L2=gradient_norm,
                    residual_last_weight_update_L2=update_norm,
                    residual_last_weight_before_SHA256=digest(last_weight_before.cpu().numpy()),
                    residual_last_weight_after_SHA256=digest(last_weight_after.cpu().numpy()),
                    residual_last_weight_gradient_SHA256=digest(last_gradient.detach().cpu().numpy()),
                    residual_last_weight_max_abs_change=float((last_weight_after-last_weight_before).abs().max().cpu()),
                    anchor_state_SHA256=anchor_digest, anchor_weights_unchanged=True,
                    initial_gate_gradient_may_be_zero=True,
                    zero_last_weight_cannot_change_from_AdamW_decay_alone=True)
                atomic(out/'first_update_gradient_review.json', first_update_evidence)
            objective += float(loss.detach().cpu()); steps += 1
        assert steps == 12 and state_digest(model.anchor.state_dict()) == anchor_digest
        _, vm = monitor(model, data['val'], tensors['val'], val_perm, guards, anchor_config)
        value = vm['conditional_pairwise_loss']
        if value < best - 1e-6:
            best, best_epoch, stale = value, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; save_best()
        else: stale += 1
        record = dict(epoch=epoch, val_conditional_loss=value, val_metrics=vm, best_epoch=best_epoch, stale_epochs=stale,
            mean_batch_training_objective=objective/steps, optimizer_steps=steps, conditional_draws=2937, regularizer_draws=2937,
            conditional_draw_sha256=digest(cd.numpy()), regularizer_draw_sha256=digest(rd.numpy()), permutation_sha256=digest(perm),
            per_source_conditional_draws=[979]*3, per_source_regularizer_draws=[979]*3,
            anchor_weights_unchanged=True, elapsed_seconds=time.perf_counter()-start)
        history.append(record); atomic(out/'train_history.json', history)
        checkpoint(out/'last.pt', dict(format=FORMAT, config=config, state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state=optimizer.state_dict(), epoch=epoch, best_state=best_state, best_loss=best, best_epoch=best_epoch,
            stale=stale, conditional_generator_state=cg.get_state(), regularizer_generator_state=rg.get_state(), history=history))
        print(json.dumps(record), flush=True)
    assert first_update_evidence is not None
    model.load_state_dict(best_state); assert state_digest(model.anchor.state_dict()) == anchor_digest
    metrics = {}
    for fold in ['train','val']:
        # Final train diagnostic uses a fixed label-free key, never a fresh fit.
        permutation = role_permutation(data[fold], variant, seed, 'val' if fold=='val' else 'train_diagnostic')
        arrays, metrics[fold] = monitor(model, data[fold], tensors[fold], permutation, guards, anchor_config)
        save_npz(out/(fold+'_components.npz'), **arrays)
    assert metrics['val']['conditional_pairwise_loss'] <= initial_metrics['conditional_pairwise_loss'] + 1e-12
    torch.cuda.synchronize(dev)
    result = dict(status='COMPLETE', protocol=pr, completed_utc=now(), variant=variant, seed=seed, actual_CUDA=True,
        checkpoint=ref(out/'model.pt'), parameters=counts, best_epoch=best_epoch, best_val_conditional_loss=best,
        initial_val_conditional_loss=initial_metrics['conditional_pairwise_loss'], initial_anchor_retained=best_epoch==-1,
        epochs_completed=len(history), optimizer_updates=12*len(history), anchor_weights_unchanged=True,
        anchor_state_sha256_before=anchor_digest, anchor_state_sha256_after=state_digest(model.anchor.state_dict()),
        metrics=metrics, elapsed_seconds=time.perf_counter()-start, peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(dev),
        environment=environment, first_update_gradient_review=ref(out/'first_update_gradient_review.json'),
        initial_pairwise_gradient_and_optimizer_update_checked=True,
        test_labels_read=False, AP_computed=False, deployment_parameters_selected=False)
    atomic(out/'metrics.json', result); atomic(out/'run_status.json', {k:result[k] for k in ['status','variant','seed','actual_CUDA','epochs_completed','optimizer_updates','anchor_weights_unchanged']})
    return result


def case_inputs(P, case):
    c = P['cases'][case]; fpath = verify(c['auxiliary_feature_manifest']); F = read(fpath)
    assert F['case'] == case and F['protocol'] == c['auxiliary_protocol'] and F['candidate_row_key_sha256'] == c['candidate_row_key_sha256']
    o = F['outputs']; arrays = {}
    for name in ['candidate_features','candidate_valid','candidate_category_id','candidate_class_order_index','fresh_material_B','query_valid','actual_supported_class_mask','supported_reference_T']:
        arrays[name] = np.load(verify(o[name+'.npy']), mmap_mode='r', allow_pickle=False)
    x, valid, B = (arrays[k] for k in ['candidate_features','candidate_valid','fresh_material_B'])
    n, C = B.shape; ci = arrays['candidate_class_order_index']; qv = arrays['query_valid']; has = arrays['actual_supported_class_mask']; T = arrays['supported_reference_T']
    assert x.shape == (n,2,13) and x.dtype == B.dtype == np.float32 and valid.dtype == qv.dtype == has.dtype == bool
    assert valid.shape == ci.shape == (n,2) and qv.shape == (n,) and has.shape == (C,)
    assert T.dtype == np.int64 and np.array_equal(T, qv.astype(np.int64)*int(has.sum()))
    assert np.all(ci[~valid] == -1) and np.all((ci[valid] >= 0) & (ci[valid] < C))
    ids = np.asarray([v['category_id'] for v in F['class_order']], np.int64)
    assert [v['class_index'] for v in F['class_order']] == list(range(C))
    assert np.array_equal(ids[ci[valid]], arrays['candidate_category_id'][valid])
    assert not np.any(x[~valid]) and np.isfinite(x).all() and np.isfinite(B).all()
    original_C = read(verify(c['C_feature_manifest'])); trace_ref = original_C['outputs']['source_trace.npz']
    with np.load(verify(trace_ref), allow_pickle=False) as z:
        keys = np.stack([z['image_id'], z['global_cache_index']], 1).astype(np.int64)
    assert len(keys) == n and len(np.unique(keys,axis=0)) == n
    return F, arrays, keys, trace_ref


def prepare_case(P, pr, case):
    actor_gate(pr)
    out = fresh(OUT/'features'/case, 'features')
    F, a, keys, trace_ref = case_inputs(P, case)
    x, valid, B, ci, qv, has, T = (a[k] for k in ['candidate_features','candidate_valid','fresh_material_B','candidate_class_order_index','query_valid','actual_supported_class_mask','supported_reference_T'])
    source = module(verify(P['sources']['N_supported_source']), '_conditional_N_supported')
    context_source = module(CODE/'src/group_data.py', '_conditional_shared_bag_context')
    n,C = B.shape; geometry = np.array(x[:,0,:8], np.float32, copy=True)
    material = np.lib.format.open_memmap(out/'full_material_columns.npy', mode='w+', dtype=np.float32, shape=(n,C,5))
    contexts = np.lib.format.open_memmap(out/'gate_context.npy', mode='w+', dtype=np.float32, shape=(n,13))
    gathered_hash = hashlib.sha256(); original_hash = hashlib.sha256()
    for lo in range(0,n,PROPOSAL_CHUNK):
        hi=min(lo+PROPOSAL_CHUNK,n); support = np.broadcast_to(has,(hi-lo,C)) & qv[lo:hi,None]
        columns,count = source.supported_columns(np.asarray(B[lo:hi]), support); assert bits(count,T[lo:hi])
        full = np.zeros((hi-lo,C,13),np.float32); full[:,:,:8]=geometry[lo:hi,None,:]
        full[:,:,8]=np.where(support,np.asarray(B[lo:hi]),np.float32(0.)); full[:,:,9:12]=columns; full[:,:,12]=support.astype(np.float32)
        material[lo:hi]=full[:,:,8:]; context,ct=context_source.gate_context(full,support,qv[lo:hi]); assert bits(ct,count)
        contexts[lo:hi]=context
        gathered=np.zeros((hi-lo,2,13),np.float32)
        for slot in [0,1]:
            local=np.flatnonzero(valid[lo:hi,slot]);rows=local+lo;gathered[local,slot]=full[local,ci[rows,slot]]
        assert bits(gathered,np.asarray(x[lo:hi])), ('fullC all13 gather mismatch',case,lo)
        gathered_hash.update(gathered.tobytes()); original_hash.update(np.asarray(x[lo:hi]).tobytes())
    material.flush();contexts.flush();del material,contexts
    save_array(out/'geometry.npy',geometry);save_array(out/'proposal_keys.npy',keys)
    proof=dict(status='PASS',protocol=pr,case=case,all13_candidate_feature_gather_bits_exact=True,
        gathered_SHA256=gathered_hash.hexdigest(),original_SHA256=original_hash.hexdigest(),
        source_auxiliary_feature_manifest=P['cases'][case]['auxiliary_feature_manifest'],
        source_class_order=F['class_order'],C=C,proposals=n,shared_gate_context_source=ref(CODE/'src/group_data.py'),
        unpermuted_context=True,new_GT_RGB_encoder_model_AP=0)
    atomic(out/'gather_proof.json',proof)
    manifest=dict(status='COMPLETE_CONDITIONAL_FULL_C_FEATURES',protocol=pr,case=case,
        source_auxiliary_feature_manifest=P['cases'][case]['auxiliary_feature_manifest'],source_C_feature_manifest=P['cases'][case]['C_feature_manifest'],
        source_trace=trace_ref,candidate_row_key_sha256=F['candidate_row_key_sha256'],class_order=F['class_order'],shape=[n,C,13],
        source_outputs=F['outputs'],outputs={name:ref(out/name)for name in ['geometry.npy','full_material_columns.npy','gate_context.npy','proposal_keys.npy']},
        gather_proof=ref(out/'gather_proof.json'),N_supported_source=P['sources']['N_supported_source'],context_source=ref(CODE/'src/group_data.py'))
    atomic(out/'feature_manifest.json',manifest);return manifest


def infer(a):
    P,pr=protocol(a.protocol,a.protocol_sha256);variant,case=a.variant,a.case
    actor_gate(pr)
    amendment, amendment_ref, continuation_review_ref = continuation_gate(pr)
    assert variant in VARIANTS and case in P['inference_cases']
    ckpath=Path(a.checkpoint).resolve();assert ckpath.is_relative_to(OUT/'models')
    ck=torch.load(ckpath,map_location='cpu',weights_only=True);config=ck['config'];seed=config['seed']
    assert ck['format']==FORMAT and config['protocol']==pr and config['variant']==variant and seed in SEEDS
    prior = read(verify(amendment['retained_inference_index']))
    assert len(prior['records']) == 18 and prior['protocol'] == pr
    assert (case,variant,seed) not in {(r['case'],r['variant'],r['seed']) for r in prior['records']}, 'Never rerun a retained completed inference'
    anchor,ac,ar=load_anchor(P,seed);assert config['anchor_checkpoint']==ar and config['anchor_config']==ac
    model=ConditionalRanker(anchor);model.load_state_dict(ck['state_dict']);assert state_digest(model.anchor.state_dict())==config['anchor_state_sha256']
    model.requires_grad_(False);dev=cuda(a.device,P);model.to(dev);model.eval()
    out=fresh(a.output_dir,'inference');assert out==OUT/'inference'/case/variant/f'seed{seed}'
    fm_path=OUT/'features'/case/'feature_manifest.json';fm=read(fm_path)
    assert fm['status']=='COMPLETE_CONDITIONAL_FULL_C_FEATURES' and fm['protocol']==pr
    assert fm['source_auxiliary_feature_manifest']==P['cases'][case]['auxiliary_feature_manifest']
    assert read(verify(fm['gather_proof']))['all13_candidate_feature_gather_bits_exact'] is True
    F,a,keys,trace_ref=case_inputs(P,case);n,C=a['fresh_material_B'].shape;valid=a['candidate_valid'];ci=a['candidate_class_order_index']
    has=a['actual_supported_class_mask'];qv=a['query_valid'];T=a['supported_reference_T']
    geometry=np.load(verify(fm['outputs']['geometry.npy']),mmap_mode='r',allow_pickle=False)
    material=np.load(verify(fm['outputs']['full_material_columns.npy']),mmap_mode='r',allow_pickle=False)
    contexts=np.load(verify(fm['outputs']['gate_context.npy']),mmap_mode='r',allow_pickle=False)
    old=P['cases'][case]['Tavailability_inference'][str(seed)];oldr=read(verify(old['receipt']))
    assert oldr['protocol']==P['sources']['T_protocol'] and oldr['checkpoint']==ar and oldr['output']==old['components']
    with np.load(verify(old['components']),allow_pickle=False)as z:oldanchor=z['full_logits'].copy()
    assert oldanchor.dtype==np.float32 and oldanchor.shape==valid.shape and not np.any(oldanchor[~valid])
    raw=np.lib.format.open_memmap(out/'full_raw_correction.npy',mode='w+',dtype=np.float32,shape=(n,C))
    centered=np.lib.format.open_memmap(out/'full_centered_correction.npy',mode='w+',dtype=np.float32,shape=(n,C))
    permutation_file=np.lib.format.open_memmap(out/'permutation_index.npy',mode='w+',dtype=np.int64,shape=(n,C))
    base=np.lib.format.open_memmap(out/'full_anchor_logits.npy',mode='w+',dtype=np.float32,shape=(n,C))
    mean=np.zeros(n,np.float64);gates=np.zeros(n,np.float32);rawg=np.zeros(valid.shape,np.float32);cg=np.zeros(valid.shape,np.float32)
    candidate_bounded=np.zeros(valid.shape,np.float32)
    max_anchor_diff=0.;anchor_bitdiff=0;max_anchor_ULP=0;minibatches=0;tuples=0;started=time.perf_counter();torch.cuda.reset_peak_memory_stats(dev)
    for lo in range(0,n,PROPOSAL_CHUNK):
        hi=min(lo+PROPOSAL_CHUNK,n);b=hi-lo;support=np.broadcast_to(has,(b,C))&qv[lo:hi,None]
        full=np.empty((b,C,13),np.float32);full[:,:,:8]=geometry[lo:hi,None,:];full[:,:,8:]=material[lo:hi]
        if variant=='permuted_material':perm=permutation_indices(support,keys[lo:hi],seed,'case:'+case)
        else:perm=np.broadcast_to(np.arange(C,dtype=np.int64),(b,C)).copy()
        permutation_file[lo:hi]=perm
        rx=full.copy();rx[:,:,8:12]=np.take_along_axis(full[:,:,8:12],perm[:,:,None],axis=1)
        cx=normalized(torch.from_numpy(np.asarray(contexts[lo:hi])).to(dev),config['guards']['context_constant_coordinates'])
        with torch.no_grad():gg=torch.sigmoid(model.gate(cx).squeeze(-1)).cpu().numpy().astype(np.float32)
        gates[lo:hi]=gg
        flat=full.reshape(-1,13);rflat=rx.reshape(-1,13);gflat=np.repeat(gg,C);sflat=support.reshape(-1)
        rawblock=np.empty(b*C,np.float32);baseblock=np.empty(b*C,np.float32);boundedblock=np.empty(b*C,np.float32)
        for start in range(0,b*C,GPU_TUPLES):
            stop=min(start+GPU_TUPLES,b*C)
            original=torch.from_numpy(flat[start:stop]).to(dev)
            xx=normalized(torch.from_numpy(rflat[start:stop]).to(dev),config['guards']['residual_constant_coordinates'])
            with torch.no_grad():
                bounded=2.*torch.tanh(model.residual(xx).squeeze(-1))
                vv=torch.from_numpy(sflat[start:stop].astype(np.float32)).to(dev)*torch.from_numpy(gflat[start:stop]).to(dev)*bounded
                bb=model.old_logits(original,ac)
            rawblock[start:stop]=vv.cpu().numpy();baseblock[start:stop]=bb.cpu().numpy();boundedblock[start:stop]=bounded.cpu().numpy();minibatches+=1;tuples+=stop-start
        rawblock=rawblock.reshape(b,C);baseblock=baseblock.reshape(b,C);boundedblock=boundedblock.reshape(b,C)
        assert np.isfinite(baseblock).all(), 'Nonfinite diagnostic frozen-anchor output'
        assert np.isfinite(rawblock).all()and np.all(np.abs(rawblock)<=2.)and not np.any(rawblock[~support])
        raw[lo:hi]=rawblock;base[lo:hi]=baseblock
        idx=np.flatnonzero(has);numerator=rawblock[:,idx].astype(np.float64).sum(1)
        mu=np.divide(numerator,T[lo:hi],out=np.zeros(b,np.float64),where=T[lo:hi]>0);mean[lo:hi]=mu
        cb=((rawblock.astype(np.float64)-mu[:,None])*.5).astype(np.float32)
        cb[~(support&(T[lo:hi,None]>1))]=0.;centered[lo:hi]=cb
        for slot in [0,1]:
            local=np.flatnonzero(valid[lo:hi,slot]);rows=local+lo;columns=ci[rows,slot]
            rawg[rows,slot]=rawblock[local,columns];cg[rows,slot]=cb[local,columns]
            candidate_bounded[rows,slot]=boundedblock[local,columns]
            freshbase=baseblock[local,columns];saved=oldanchor[rows,slot]
            max_anchor_diff=max(max_anchor_diff,float(np.max(np.abs(freshbase.astype(np.float64)-saved.astype(np.float64)),initial=0.)))
            anchor_bitdiff+=int(np.count_nonzero(freshbase.view(np.uint32)!=saved.view(np.uint32)))
            max_anchor_ULP=max(max_anchor_ULP,int(np.max(float32_ULP_distance(freshbase,saved),initial=0)))
        # Fresh fullC anchor is unused by deployment. Its finite error/ULP is reported,
        # not used to accept material corrections or a test result. Saved anchor and
        # full1355 frozen state stay exact under the separately frozen amendment.
        atomic(out/'progress.json',dict(status='ACTUAL_CONDITIONAL_FULL_C_GPU_INFERENCE',protocol=pr,case=case,variant=variant,seed=seed,
            completed_proposals=hi,total_proposals=n,actual_GPU_model_forward_minibatches=minibatches,actual_model_input_tuples=tuples))
    raw.flush();centered.flush();permutation_file.flush();base.flush();del raw,centered,permutation_file,base
    availability=np.asarray(a['candidate_features'][:,:,12],np.float32).copy()
    assert state_digest(model.anchor.state_dict())==config['anchor_state_sha256']
    save_npz(out/'components.npz',anchor_logits=oldanchor,raw_candidate_correction=rawg,supported_mean=mean,
        centered_correction=cg,availability=availability,valid=np.asarray(valid),candidate_class_order_index=np.asarray(ci),supported_count=np.asarray(T),gate_weights=gates,
        candidate_bounded_residual=candidate_bounded)
    torch.cuda.synchronize(dev)
    result=dict(status='COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE',protocol=pr,protocol_sha256=pr['sha256'],completed_utc=now(),
        case=case,variant=variant,seed=seed,checkpoint=ref(ckpath),parameters=config['parameters'],feature_manifest=ref(fm_path),
        source_auxiliary_feature_manifest=P['cases'][case]['auxiliary_feature_manifest'],source_C_feature_manifest=P['cases'][case]['C_feature_manifest'],
        candidate_row_key_sha256=F['candidate_row_key_sha256'],Tavailability_inference=old,anchor_checkpoint=ar,
        output=ref(out/'components.npz'),full_raw_correction=ref(out/'full_raw_correction.npy'),full_centered_correction=ref(out/'full_centered_correction.npy'),
        full_anchor_logits=ref(out/'full_anchor_logits.npy'),permutation_index=ref(out/'permutation_index.npy'),
        proposal_context_shared_unpermuted=True,full_C=C,proposals=n,shape=list(valid.shape),dtype='float32',
        actual_GPU_model_forward_minibatches=minibatches,actual_model_input_tuples=tuples,actual_common_gate_proposal_forwards=n,
        actual_CUDA=True,new_GPU_inference=True,new_training=0,reference_reused=False,
        new_delta_unsupported_and_T_le1_exact0=True,candidate_anchor_saved_Tfull_logits_exact=True,
        fresh_fullC_anchor_candidate_comparison=dict(max_absdiff=max_anchor_diff,F32_bit_differences=anchor_bitdiff,max_ULP_difference=max_anchor_ULP,finite=True,diagnostic_only=True,absolute_gate_removed_by_verification_amendment=True,original_atol=1e-6,bitexact_claimed=False),
        verification_amendment=amendment_ref,continued_model_preflight_review=continuation_review_ref,
        continued_inference_after_diagnostic_guard_only=True,inference_implementation=ref(__file__),
        frozen_T1355_state_digest=config['anchor_state_sha256'],
        center_precision='Full truly supported canonical class subset CPU F64 sum/T, difference/2 then F32',
        current_trainable_parameters=0,source_training_parameters=362,frozen_parameters_during_inference=1717,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],GPU_name=torch.cuda.get_device_name(dev),PID=os.getpid(),argv=sys.argv,
        elapsed_seconds=time.perf_counter()-started,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(dev),
        GT_opened=False,labels_read=False,AP_computed=False,deployment_parameters_selected=False,
        deployment_quality_not_producer_sigmoid=True,zero_delta_fallback='Verified formal T availability deployed per-seed quality; endpoint-owned exact reconstruction')
    atomic(out/'inference.json',result);return result
