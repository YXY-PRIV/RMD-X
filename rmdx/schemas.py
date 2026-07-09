"""Data contracts used by RMD-X wrapper modules.

These contracts describe the files produced by RAXO. They do not implement
reliability, material, or risk scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PrototypeRecord:
    key: str
    supercategory: str
    feature_dim: int
    tensor_shape: tuple[int, ...]
    dtype: str
    is_negative: bool


@dataclass(frozen=True)
class ProposalRecord:
    image_id: int
    annotation_id: Optional[int]
    category_id: int
    bbox: list[float]
    score: float
    has_mask: bool
    has_segmentation: bool


@dataclass(frozen=True)
class DetectionRecord:
    image_id: int
    category_id: int
    bbox: list[float]
    score: float
    dist_cosine_inter: Optional[float]
    dist_cosine_intra: Optional[float]


@dataclass(frozen=True)
class CategoryRecord:
    id: int
    name: str


@dataclass(frozen=True)
class RMDXScoreRecord:
    image_id: int
    bbox: list[float]
    original_category_id: int
    rmdx_category_id: int
    original_score: float
    score_rmdx: float
    reliability_score: Optional[float] = None
    material_score: Optional[float] = None
    risk_score: Optional[float] = None

