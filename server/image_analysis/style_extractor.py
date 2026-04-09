# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Style extraction and consistency enforcement for image-conditioned scene generation.

This module extracts style descriptors from VLM scene analysis and provides
constraints that guide object retrieval and TRELLIS 3D generation to maintain
visual consistency with the reference image.

Style information is injected into the TRELLIS text captions and object
descriptions — it does NOT replace the real TRELLIS/MatFuse pipelines.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class StyleDescriptor:
    """Describes the visual style extracted from a reference image.

    Attributes:
        color_palette: Dominant colours as hex strings (e.g. ``["#8B4513", "#F5F5DC"]``).
        material_palette: Dominant materials (e.g. ``["oak wood", "brushed steel"]``).
        era: Temporal style era (e.g. ``"mid-century modern"``).
        aesthetic_keywords: Free-form style keywords (e.g. ``["minimalist", "warm"]``).
        dominant_colors: Top 3 colour names for quick matching (e.g. ``["brown", "beige"]``).
    """

    color_palette: List[str] = field(default_factory=list)
    material_palette: List[str] = field(default_factory=list)
    era: str = ""
    aesthetic_keywords: List[str] = field(default_factory=list)
    dominant_colors: List[str] = field(default_factory=list)


class StyleConstraint:
    """Scores how well an object description matches a target style.

    This is used to *boost* descriptions fed to TRELLIS so that the generated
    3D assets are stylistically consistent with the reference image.  It does
    **not** replace TRELLIS or MatFuse — it enriches the text prompts that
    drive them.
    """

    def __init__(self, descriptor: StyleDescriptor) -> None:
        self.descriptor = descriptor
        # Build a flat set of keywords for fast overlap scoring
        self._keywords: set[str] = set()
        for kw in descriptor.aesthetic_keywords:
            self._keywords.update(kw.lower().split())
        for mat in descriptor.material_palette:
            self._keywords.update(mat.lower().split())
        if descriptor.era:
            self._keywords.update(descriptor.era.lower().split())
        for col in descriptor.dominant_colors:
            self._keywords.update(col.lower().split())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_match(self, object_description: str) -> float:
        """Return a 0-1 score indicating how well *object_description* fits the style.

        Uses keyword overlap between the style descriptor and the description.
        A higher score means more keywords from the style appear in the text.
        """
        if not self._keywords or not object_description:
            return 0.0
        desc_tokens = set(object_description.lower().split())
        overlap = len(self._keywords & desc_tokens)
        return min(1.0, overlap / max(1, len(self._keywords)))

    # ------------------------------------------------------------------
    # Caption enrichment (the main integration point with TRELLIS)
    # ------------------------------------------------------------------

    def enrich_caption(self, base_caption: str) -> str:
        """Append concise style hints to a TRELLIS generation caption.

        The returned string keeps the original caption intact and adds a
        short style suffix so that TRELLIS produces assets in the right
        aesthetic without altering the object identity.

        Example::

            >>> sc.enrich_caption("A wooden office chair")
            "A wooden office chair, mid-century modern style, oak wood, warm tones"
        """
        parts: list[str] = []
        if self.descriptor.era:
            parts.append(f"{self.descriptor.era} style")
        # Pick at most 2 materials to keep the caption short
        for mat in self.descriptor.material_palette[:2]:
            parts.append(mat)
        # Pick at most 2 dominant colour names
        for col in self.descriptor.dominant_colors[:2]:
            parts.append(f"{col} tones")
        # Add up to 2 aesthetic keywords
        for kw in self.descriptor.aesthetic_keywords[:2]:
            if kw.lower() not in base_caption.lower():
                parts.append(kw)
        if not parts:
            return base_caption
        return f"{base_caption}, {', '.join(parts)}"


def extract_style(analysis: "SceneAnalysis") -> StyleDescriptor:  # noqa: F821 – forward ref
    """Derive a :class:`StyleDescriptor` from a VLM :class:`SceneAnalysis`.

    The function inspects the ``style``, ``vibe``, and per-object attributes
    already present in the analysis (which came from the real VLM call) and
    distils them into a compact descriptor.

    Args:
        analysis: A :class:`~image_analysis.scene_analyzer.SceneAnalysis`
                  produced by the VLM scene analyser.

    Returns:
        A populated :class:`StyleDescriptor`.
    """
    desc = StyleDescriptor()

    # --- era ---
    vibe = analysis.vibe or {}
    desc.era = vibe.get("era", "") or ""

    # --- aesthetic keywords from style string + vibe ---
    if analysis.style:
        desc.aesthetic_keywords.extend(analysis.style.lower().split())
    for key in ("mood", "lighting"):
        val = vibe.get(key, "")
        if val:
            desc.aesthetic_keywords.append(val.lower())

    # --- materials & colours from objects ---
    material_counts: dict[str, int] = {}
    color_words: dict[str, int] = {}
    _colour_names = {
        "red", "orange", "yellow", "green", "blue", "purple", "pink",
        "brown", "black", "white", "grey", "gray", "beige", "ivory",
        "tan", "cream", "gold", "silver", "bronze", "teal", "navy",
        "maroon", "olive", "coral", "turquoise", "charcoal",
    }
    for obj in analysis.objects:
        mat = (obj.material or "").strip().lower()
        if mat:
            material_counts[mat] = material_counts.get(mat, 0) + obj.quantity
        # Extract colour words from the description
        for token in (obj.description or "").lower().split():
            clean = token.strip(".,;:!?\"'()[]")
            if clean in _colour_names:
                color_words[clean] = color_words.get(clean, 0) + 1

    # Top materials
    desc.material_palette = [
        m for m, _ in sorted(material_counts.items(), key=lambda x: -x[1])
    ][:5]

    # Top colours
    desc.dominant_colors = [
        c for c, _ in sorted(color_words.items(), key=lambda x: -x[1])
    ][:5]

    return desc
