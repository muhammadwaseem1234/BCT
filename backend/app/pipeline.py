from pathlib import Path
from typing import Any, Dict

from app.classifiers.block_classifier import classify_subsection
from app.hierarchy.section_detector import detect_sections
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.span_extractor import extract_spans
from app.layout.block_builder import build_blocks
from app.layout.x_clusterer import cluster_x_positions
from app.layout.y_clusterer import cluster_y_positions


def parse_document(path: str, debug: bool = False) -> Dict[str, Any]:
    """End-to-end parser: PDF -> spans -> blocks -> hierarchy -> classification -> JSON."""
    doc = load_pdf(path)
    spans = extract_spans(doc)
    blocks = build_blocks(spans)

    grouped_sections, section_debug = detect_sections(blocks)

    output_sections = []
    for section in grouped_sections:
        subsection_outputs = []
        for subsection in section["subsections"]:
            parsed_subsection, _ = classify_subsection(subsection)
            if parsed_subsection["type"] == "paragraph" and not parsed_subsection.get("content", "").strip():
                continue
            subsection_outputs.append(parsed_subsection)

        if not subsection_outputs:
            subsection_outputs.append(
                {
                    "title": "Overview",
                    "type": "paragraph",
                    "content": "",
                }
            )

        output_sections.append({"title": section["title"], "subsections": subsection_outputs})

    document_title = output_sections[0]["title"] if output_sections else Path(path).stem
    response: Dict[str, Any] = {
        "document_title": document_title,
        "sections": output_sections,
    }

    if debug:
        response["debug"] = {
            "raw_blocks": blocks,
            "detected_columns": cluster_x_positions(blocks),
            "detected_rows": cluster_y_positions(blocks),
            "section_debug": section_debug,
        }

    return response
