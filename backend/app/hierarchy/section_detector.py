from collections import Counter
from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

from app.layout.gap_analyzer import has_large_vertical_gap


@dataclass
class HeadingThresholds:
    body_size: float
    h1_size: float
    h2_size: float


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w])


def _is_heading_text(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    if clean.endswith((".", ":", ";")):
        return False
    return _word_count(clean) <= 12


def _compute_thresholds(blocks: List[dict]) -> HeadingThresholds:
    sizes = [round(float(b["font_size"]), 1) for b in blocks if b.get("text", "").strip()]
    if not sizes:
        return HeadingThresholds(body_size=10.0, h1_size=12.0, h2_size=11.0)

    body_size = Counter(sizes).most_common(1)[0][0]
    larger = sorted([s for s in set(sizes) if s > body_size], reverse=True)

    h1_size = larger[0] if larger else body_size
    h2_size = larger[1] if len(larger) > 1 else (larger[0] if larger else body_size)
    return HeadingThresholds(body_size=float(body_size), h1_size=float(h1_size), h2_size=float(h2_size))


def _heading_level(block: dict, thresholds: HeadingThresholds, prev_block: Optional[dict]) -> int:
    text = block.get("text", "").strip()
    if not _is_heading_text(text):
        return 0

    if re.match(r"^\d+(?:\.\d+)*[\.)]?\s+", text):
        return 2

    size = float(block.get("font_size", 0.0))
    bold_ratio = float(block.get("bold_ratio", 0.0))
    big_gap = bool(prev_block and has_large_vertical_gap(prev_block, block, threshold=max(8.0, thresholds.body_size)))

    if size >= thresholds.h1_size and (bold_ratio >= 0.25 or big_gap):
        return 1
    if size >= thresholds.h2_size and (bold_ratio >= 0.2 or big_gap):
        return 2
    return 0


def detect_sections(blocks: List[dict]) -> Tuple[List[Dict], Dict]:
    """Build section/subsection groups from line blocks using font and spacing signals."""
    thresholds = _compute_thresholds(blocks)
    sections: List[Dict] = []
    debug_headings: List[Dict] = []

    current_section: Optional[Dict] = None
    current_subsection: Optional[Dict] = None

    for idx, block in enumerate(blocks):
        prev = blocks[idx - 1] if idx > 0 else None
        level = _heading_level(block, thresholds, prev)

        if level == 1:
            current_section = {"title": block["text"], "subsections": []}
            sections.append(current_section)
            current_subsection = {"title": "Overview", "lines": []}
            current_section["subsections"].append(current_subsection)
            debug_headings.append({"text": block["text"], "level": "H1"})
            continue

        if level == 2:
            if current_section is None:
                current_section = {"title": "Document", "subsections": []}
                sections.append(current_section)
            current_subsection = {"title": block["text"], "lines": []}
            current_section["subsections"].append(current_subsection)
            debug_headings.append({"text": block["text"], "level": "H2"})
            continue

        if current_section is None:
            current_section = {"title": "Document", "subsections": []}
            sections.append(current_section)

        if current_subsection is None:
            current_subsection = {"title": "Overview", "lines": []}
            current_section["subsections"].append(current_subsection)

        current_subsection["lines"].append(block)

    if not sections:
        sections = [{"title": "Document", "subsections": [{"title": "Overview", "lines": blocks}]}]

    debug = {
        "heading_thresholds": {
            "body_size": thresholds.body_size,
            "h1_size": thresholds.h1_size,
            "h2_size": thresholds.h2_size,
        },
        "detected_headings": debug_headings,
    }
    return sections, debug
