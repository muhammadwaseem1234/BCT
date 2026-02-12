import re
from typing import Dict, List, Tuple

from app.table_engine.grid_detector import detect_table

BULLET_PATTERN = re.compile(r"^(?:[-*•]|\d+[\.)]|[A-Za-z][\.)])\s+")


def _extract_bullets(lines: List[dict]) -> List[str]:
    items: List[str] = []
    for line in lines:
        text = line.get("text", "").strip()
        if BULLET_PATTERN.match(text):
            items.append(BULLET_PATTERN.sub("", text).strip())
    return [i for i in items if i]


def classify_subsection(subsection: Dict) -> Tuple[Dict, Dict]:
    """Classify subsection content as paragraph, bullet_list, or table."""
    lines = subsection.get("lines", [])
    title = subsection.get("title", "Untitled")

    bullet_items = _extract_bullets(lines)
    if len(bullet_items) >= 2 and len(bullet_items) >= max(2, int(len(lines) * 0.5)):
        return (
            {
                "title": title,
                "type": "bullet_list",
                "items": bullet_items,
            },
            {"classification": "bullet_list"},
        )

    table = detect_table(lines)
    if table:
        return (
            {
                "title": title,
                "type": "table",
                "headers": table["headers"],
                "rows": table["rows"],
            },
            {
                "classification": "table",
                "table_columns": table.get("detected_columns", []),
                "table_rows": table.get("detected_rows", []),
            },
        )

    paragraph = "\n".join(line.get("text", "").strip() for line in lines if line.get("text", "").strip())
    return (
        {
            "title": title,
            "type": "paragraph",
            "content": paragraph,
        },
        {"classification": "paragraph"},
    )
