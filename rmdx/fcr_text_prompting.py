"""Prompt helpers for OpenCLIP FCR-RMD-X."""

from __future__ import annotations

import re

PROMPT_TEMPLATES = [
    "an x-ray image of a {category}",
    "a security x-ray image containing a {category}",
    "an x-ray scan showing a {category}",
    "a cropped x-ray object of {category}",
    "{category}",
]


def normalize_category_name(name: str) -> str:
    value = str(name).replace("_", " ").strip().lower()
    value = re.sub(r"[^a-z0-9\- ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def prompts_for_category(name: str) -> list[str]:
    clean = normalize_category_name(name)
    return [template.format(category=clean) for template in PROMPT_TEMPLATES]

