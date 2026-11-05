# Current RMD-X source and recorded evidence

This directory accompanies **Conflict-Aware Appearance Fusion for Open-Vocabulary X-Ray Object Detection**. It publishes the selected conflict-aware method, its scientific source dependencies, small author-trained quality checkpoints, and recorded numerical results. It does not run training, inference or evaluation on import.

## Start with inspection

```bash
python reproduction/inspect_release.py
python reproduction/tests/test_pair_contract.py
```

The first command uses only the Python standard library and checks the supplied file inventory. It does not load a model or a provider dataset, evaluate AP, run bootstrap, create predictions, or download resources. The second needs NumPy and checks small artificial counterexamples for signed pair correction, support/branch fallback, legal category ID zero, slot mapping, fixed-anchor tie choice, clipping/mixing and deterministic Gaussian suppression. These checks are not experimental AP results or a from-scratch scientific reproduction.

## Two different source roles

- `portable/core.py` contains the recorded `project`, quality mixing, branch selection and Gaussian functions extracted **verbatim** into an import-safe module. `portable/function_origins.json` identifies their source-file and exact function hashes. This adapter accepts arrays supplied by its caller; it is not a detector or a complete quality-head trainer.
- `recorded/experiments/` preserves the actual scientific modules and selected preparation/training/inference/evaluation scripts. `recorded/new_module_evidence/` preserves the CI, fixed-weight ablation and runtime implementations. Their files retain their original bytes and SHA identities. Historical absolute resource paths are retained as provenance inside these recorded snapshots; **they are not the configurable portable entry point**. Their frozen protocols and native SHA gates must not be silently rewritten to make a new filesystem look like an independently reproduced original run.

`resources.example.json` documents configurable mount roots for resource inspection. Set those environment variables and pass a local edited copy with `--resources`. It checks path existence only. Full original training or AP replay still needs the corresponding provider annotations/images, original feature manifests/cache tensors, prediction inputs, native evaluator dependencies and exact identities. No claim is made that editing this example alone makes every recorded script runnable on a new machine.

## Source entry points

| Concern | Recorded source |
|---|---|
| Signed zero-sum pair projection | `recorded/experiments/pair_conflict_material_projection_20261101/src/pair_projection.py` |
| Fixed branch and final endpoint export | `recorded/experiments/pair_conflict_material_projection_20261101/src/pair_endpoint.py` |
| Exact deployed-anchor reconstruction | `recorded/experiments/pair_conflict_material_projection_20261101/src/anchor_reference.py` |
| Learned conditional residual | `recorded/experiments/material_conditional_ranking_fix1_20261101/src/quality_model.py` |
| Paired CI gates and replay | `recorded/new_module_evidence/ci/scripts/run_ci.py` |
| Four fixed-weight component cells | `recorded/new_module_evidence/ablation/scripts/run_dataset.py` |
| Complete learned-quality deployment | `recorded/new_module_evidence/runtime/src/deployment.py` |
| Fresh image-to-output measurement | `recorded/new_module_evidence/runtime/scripts/run_raw_benchmark.py` |

The two pure endpoint dependencies missing from the earlier compact archive (`appearance_quality_reranking.../quality_endpoint.py` and `appearance_candidate_refinement.../candidate_endpoint.py`) are now included from the unchanged local originals. The original-quality model and ensemble endpoint are also supplied. The twelve checkpoint archives (three selected conditional heads, three permuted controls, three availability anchors and three visual-clean heads) are small author-trained quality heads, not the large upstream detector or image encoder. They are published byte-for-byte, including their recorded configuration/identity metadata; inspect hashes before loading and use an appropriate `torch.load(..., weights_only=True)` workflow.

## Results and limits

`results/latest/` contains the complete reported logical endpoint table, primary comparisons and diagnostics. `results/ci/`, `results/ablation/` and `results/runtime/` preserve the recorded uncertainty, component and latency/memory evidence. The paired CI vectors and endpoint replicates are small numerical AP results; they are not images, detector predictions or training feature tensors.

Candidate union and visual candidates use the shared frozen scorer. The final conflict correction has a small mean gain over union; two dataset intervals include zero. The fixed-weight component study has both positive and negative cells. The earlier material-only transfer regressions are not erased by this publication. CIs are conditional on the retained datasets/predictions and do not include all training/support-selection variance. The raw-image timing cohort is a separate recorded workload; its upstream identity with the historical cached AP is not certified. Cached quality cost must not be presented as complete image inference time. The final branch, category and box remain anchor-fixed; D can have either sign, and zero sum is a logit property, not probability calibration or a claim that F always improves.

Provider radiographs, datasets/GT releases, large feature/prediction caches, detector/encoder weights, private correspondence and author submission declarations are not distributed here. Obtain external resources through their providers under the applicable terms. No original helper/protocol hash is changed to conceal relocation. `SOURCE_MANIFEST.json` distinguishes exact recorded bytes from the import-safe adapter and publication metadata. No scientific experiment was rerun when preparing this public snapshot.
