# Datasets and External Resources

This repository is a code release. It does not redistribute X-ray datasets,
model checkpoints, detector proposals, or precomputed caches. Prepare those
resources from their original providers and place them under the paths below,
or edit the scripts to match your local layout.

## 1. Recommended Mount Layout

The released scripts were developed with the following container paths:

```text
/workspace/RAXO
/models/ICCV25_experimentation
/datasets/xray-datasets
```

Recommended Docker-style mounts:

```text
/path/to/RMD-X                         -> /workspace/RAXO
/path/to/ICCV25_experimentation        -> /models/ICCV25_experimentation
/path/to/xray-datasets                 -> /datasets/xray-datasets
```

## 2. Dataset Layout

All datasets are expected to use a COCO-format test annotation named
`full_test.json` and a test image directory named `test`.

```text
/datasets/xray-datasets/
├── DET-COMPASS/
│   ├── annotations/full_test.json
│   └── test/
├── PIXray/
│   ├── annotations/full_test.json
│   └── test/
├── pidray/
│   ├── annotations/full_test.json
│   └── test/
├── CLCXray/
│   ├── annotations/full_test.json
│   └── test/
├── HiXray/
│   ├── annotations/full_test.json
│   └── test/
└── DvXray/
    ├── annotations/full_test.json
    └── test/
```

The expected dataset names in scripts are case-sensitive. If your local PIDray
folder is named `PIDray`, either rename the folder or update the corresponding
script constant.

## 3. RAXO Prototype and Detector Proposal Layout

```text
/models/ICCV25_experimentation/
├── known_prototypes_sam2.pt
└── initial_detections/
    ├── groundingDINO/
    ├── VLDet/
    └── CoDet/
```

For each detector and dataset, the expected with-mask proposal file is:

```text
/models/ICCV25_experimentation/initial_detections/<detector>/coco_results.bbox_<dataset>_with_masks.json
```

For initial detector AP evaluation, the expected proposal file is:

```text
/models/ICCV25_experimentation/initial_detections/<detector>/coco_results.bbox_<dataset>.json
```

The DET-COMPASS GroundingDINO initial result used in the authors' experiments
was recovered from the official with-mask file because the raw JSON resource was
truncated. The recovered file name used by the scripts is:

```text
coco_results.bbox_DET-COMPASS.recovered_from_with_masks.json
```

## 4. Main RMD-X Material Bank

The locked paper result uses the support-derived colour-only material bank:

```text
/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json
```

This bank should be built from non-test support/prototype data. Do not build the
main material bank from `full_test.json` ground-truth boxes.

## 5. Main RMD-X Cache

The cache-based reproduction uses:

```text
/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full/
```

Typical cache files:

```text
visual_similarity.pt
material_class_similarity.pt
proposal_meta.jsonl
prototype_meta.json
category_meta.json
cache_summary.json
```

Large cache files are not tracked by Git. Rebuild them locally from the prepared
resources.

## 6. Official / External Resources to Prepare

The following resources must be obtained externally:

| Resource | Purpose | Included in this repo |
|---|---|---|
| DET-COMPASS test images and annotation | Main benchmark | No |
| PIXray / PIDray / CLCXray / HiXray / DvXray test images and annotations | Multi-dataset evaluation | No |
| `known_prototypes_sam2.pt` | RAXO in-house visual descriptors | No |
| GroundingDINO / VLDet / CoDet proposal files | RAXO and RMD-X input proposals | No |
| with-mask proposal JSON files | RAXO known branch and RMD-X cache input | No |
| DINOv2 / SAM2 / detector checkpoints | Original RAXO runtime dependencies | No |
| RMD-X generated caches | Reproducibility acceleration | No |

## 7. Minimal DET-COMPASS Checklist

Before running the main DET-COMPASS experiment, verify:

```bash
test -f /models/ICCV25_experimentation/known_prototypes_sam2.pt
test -f /models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_DET-COMPASS_with_masks.json
test -f /datasets/xray-datasets/DET-COMPASS/annotations/full_test.json
test -d /datasets/xray-datasets/DET-COMPASS/test
```

Then check the code/data contract:

```bash
cd /workspace/RAXO
python3 scripts_rmdx/03_check_rmdx_contracts.py
```

## 8. Notes on Detector Coverage

The paper reports DET-COMPASS multi-detector evidence for GroundingDINO, VLDet,
and CoDet. Detic is excluded from the DET-COMPASS main multi-detector experiment
because official DET-COMPASS Detic proposal resources were not available.

