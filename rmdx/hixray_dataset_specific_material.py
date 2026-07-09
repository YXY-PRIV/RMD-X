"""HiXray dataset-specific material helper wrappers for Batch38.

This module is external to RAXO. It reuses the generic dataset-specific
material similarity builder with the HiXray train-only material bank.
"""

from __future__ import annotations

from rmdx.dataset_specific_material import (  # noqa: F401
    build_dataset_specific_material_similarity,
    extract_dataset_specific_proposal_features,
    load_material_bank,
)

