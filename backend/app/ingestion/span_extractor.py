from typing import Any, Dict, List


def extract_spans(doc) -> List[Dict[str, Any]]:
    """Extract text spans from each page with geometry and style metadata."""
    spans: List[Dict[str, Any]] = []

    for page_number, page in enumerate(doc):
        page_dict = page.get_text("dict", sort=True)

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    bbox = tuple(span.get("bbox", (0, 0, 0, 0)))
                    y0 = float(bbox[1])
                    y1 = float(bbox[3])

                    # Remove top and bottom page bands (headers/footers).
                    if y0 < 90 or y1 > 740:
                        continue

                    spans.append(
                        {
                            "text": text,
                            "size": float(span.get("size", 0.0)),
                            "font": str(span.get("font", "")),
                            "bbox": bbox,
                            "page": page_number,
                        }
                    )

    return spans
