# Reproduction Guide

This guide gives the intended entry points for reproducing the RMD-X results
after all external resources have been prepared.

## 1. Environment

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

If using Docker, use the mount layout documented in
`DATASETS_AND_RESOURCES.md`.

## 2. Contract Check

Run the contract checker first:

```bash
cd /workspace/RAXO
python3 scripts_rmdx/03_check_rmdx_contracts.py
```

Fix missing paths before running expensive stages.

## 3. Main DET-COMPASS + GroundingDINO Result

Build the support-derived material bank:

```bash
python3 scripts_rmdx/06_build_material_colour_bank.py
```

Build the DET-COMPASS cache:

```bash
python3 scripts_rmdx/15_build_preargmax_cache_full.py
```

Run the fixed material pre-argmax fusion grid:

```bash
python3 scripts_rmdx/16_run_cache_fusion_grid.py
```

Validate reproduction:

```bash
python3 scripts_rmdx/17_validate_cache_reproduction.py
python3 scripts_rmdx/19_run_cache_regression_test.py
```

Expected locked reference:

```text
visual_only RAXO reproduction: AP / AP50 / AP75 = 0.480 / 0.544 / 0.488
RMD-X fixed material result:   AP / AP50 / AP75 = 0.501 / 0.561 / 0.515
```

## 4. Multi-detector Experiments

Check detector resources:

```bash
python3 scripts_rmdx/20_check_multidetector_resources.py
python3 scripts_rmdx/21_summarize_multidetector_readiness.py
```

Run or summarize RAXO baselines:

```bash
python3 scripts_rmdx/24_run_multidetector_raxo_baseline.py
python3 scripts_rmdx/25_summarize_multidetector_baseline.py
```

Build detector-specific caches and run RMD-X fusion:

```bash
python3 scripts_rmdx/26_build_multidetector_preargmax_cache.py
python3 scripts_rmdx/27_run_multidetector_cache_fusion_grid.py
python3 scripts_rmdx/28_summarize_multidetector_rmdx.py
```

## 5. Multi-dataset Experiments

Check dataset resources:

```bash
python3 scripts_rmdx/29_check_multidataset_resources.py
python3 scripts_rmdx/30_summarize_multidataset_readiness.py
```

Run or summarize RAXO baselines:

```bash
python3 scripts_rmdx/38_run_multidataset_raxo_baseline.py
python3 scripts_rmdx/39_summarize_multidataset_raxo_baseline.py
```

Build transfer material banks and run RMD-X:

```bash
python3 scripts_rmdx/45_inspect_multidataset_category_mapping.py
python3 scripts_rmdx/46_discover_multidataset_support_sources.py
python3 scripts_rmdx/47_build_multidataset_transfer_material_banks.py
python3 scripts_rmdx/48_validate_multidataset_material_banks.py
python3 scripts_rmdx/49_build_multidataset_preargmax_cache_parallel.py
python3 scripts_rmdx/50_run_multidataset_cache_fusion_grid.py
python3 scripts_rmdx/51_summarize_multidataset_rmdx_results.py
```

## 6. Optional Ablations

The repository also includes scripts for multi-channel material banks, learned
material adapters, adaptive gates, and diagnostic variants. These are included
for transparency and ablation reproduction, but the locked main paper method is
the fixed colour-only pre-argmax fusion.

## 7. Output Policy

All generated artifacts should be written under `outputs_rmdx/`. This directory
is ignored by Git except for `.gitkeep`, so large caches and logs will not be
accidentally committed.

