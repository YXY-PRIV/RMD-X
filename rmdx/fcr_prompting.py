"""Prompt helpers for FCR-RMD-X."""

from __future__ import annotations

import re


PROMPT_TEMPLATES = [
    "an x-ray image of a {category}",
    "a security x-ray image containing {category}",
    "a prohibited item: {category}",
    "a cropped x-ray object of {category}",
    "{category}",
]


def normalize_category_name(name: str) -> str:
    value = str(name).replace("_", " ").lower()
    value = re.sub(r"[^a-z0-9\- ]+", " ", value)
    value = re.sub(r"\\s+", " ", value).strip()
    return value


def prompts_for_category(name: str) -> list[str]:
    clean = normalize_category_name(name)
    return [template.format(category=clean) for template in PROMPT_TEMPLATES]
