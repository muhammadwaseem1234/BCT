from statistics import median
from typing import Any, Dict, List, Tuple


def _merge_bbox(bboxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def build_blocks(spans: List[Dict[str, Any]], y_tolerance: float = 2.5) -> List[Dict[str, Any]]:
    """Group spans into line-level blocks while preserving atoms for table detection."""
    grouped: Dict[Tuple[int, float], List[Dict[str, Any]]] = {}

    for span in spans:
        y0 = float(span["bbox"][1])
        page = int(span["page"])
        line_key = round(y0 / y_tolerance) * y_tolerance
        key = (page, line_key)
        grouped.setdefault(key, []).append(span)

    blocks: List[Dict[str, Any]] = []
    for (page, _), line_spans in grouped.items():
        ordered = sorted(line_spans, key=lambda s: s["bbox"][0])
        text = " ".join(s["text"] for s in ordered).strip()
        if not text:
            continue

        bboxes = [tuple(s["bbox"]) for s in ordered]
        sizes = [float(s["size"]) for s in ordered]
        bold_count = sum(1 for s in ordered if "bold" in str(s["font"]).lower())

        blocks.append(
            {
                "text": text,
                "font_size": float(median(sizes)) if sizes else 0.0,
                "bold_ratio": (bold_count / len(ordered)) if ordered else 0.0,
                "bold": bold_count > 0,
                "bbox": _merge_bbox(bboxes),
                "page": page,
                "atoms": [
                    {
                        "text": s["text"],
                        "bbox": tuple(s["bbox"]),
                        "bold": "bold" in str(s["font"]).lower(),
                    }
                    for s in ordered
                ],
            }
        )

    blocks.sort(key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))
    return blocks
