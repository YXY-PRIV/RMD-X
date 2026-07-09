# Data and Resource Contract

This document describes the resource layout expected by the released scripts.
The datasets, model weights, and precomputed detections are **not** included in
this GitHub package.

## Recommended Container Mounts

```text
/path/to/RMD_X_GitHub_upload                  -> /workspace/RAXO
/path/to/models/ICCV25_experimentation       -> /models/ICCV25_experimentation
/path/to/datasets/xray-datasets              -> /datasets/xray-datasets
```

## Core Paths

```text
/models/ICCV25_experimentation/
├── known_prototypes_sam2.pt
└── initial_detections/
    └── groundingDINO/
        ├── coco_results.bbox_DET-COMPASS.json
        ├── coco_results.bbox_DET-COMPASS_with_masks.json
        ├── coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final.json
        └── coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final_uncertanty.json

/datasets/xray-datasets/
└── DET-COMPASS/
    ├── annotations/
    │   └── full_test.json
    └── test/
```

## Main RMD-X Resources

The fixed colour-only material bank used by the main result is expected at:

```text
/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json
```

The full DET-COMPASS RMD-X cache is expected at:

```text
/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full/
```

Cache contents:

```text
visual_similarity.pt
material_class_similarity.pt
proposal_meta.jsonl
prototype_meta.json
category_meta.json
cache_summary.json
```

## Multi-detector Resources

For multi-detector experiments, the same DET-COMPASS annotation and image root
are used, with detector-specific with-mask files:

```text
/models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_DET-COMPASS_with_masks.json
/models/ICCV25_experimentation/initial_detections/VLDet/coco_results.bbox_DET-COMPASS_with_masks.json
/models/ICCV25_experimentation/initial_detections/CoDet/coco_results.bbox_DET-COMPASS_with_masks.json
```

The paper excludes Detic on DET-COMPASS because official Detic proposals were
not available.

## Multi-dataset Resources

For multi-dataset experiments, the expected pattern is:

```text
/datasets/xray-datasets/<dataset>/annotations/full_test.json
/datasets/xray-datasets/<dataset>/test/
/models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_<dataset>_with_masks.json
```

Datasets used in the paper include:

```text
DET-COMPASS
PIXray
pidray
CLCXray
HiXray
DvXray
```

## Leakage Policy

The main material bank should be built from non-test support/prototype data. Do
not use `full_test.json` ground-truth boxes to construct the main material bank.

## More Details

See `DATASETS_AND_RESOURCES.md` for a more explicit checklist of dataset names,
proposal file names, with-mask resources, material banks, and cache outputs.
