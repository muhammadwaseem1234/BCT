from typing import Dict, List


def _is_numeric(text: str) -> bool:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "").strip()
    if not cleaned:
        return False
    if cleaned.count(".") <= 1 and cleaned.replace(".", "", 1).isdigit():
        return True
    return False


def detect_headers(rows: List[List[Dict]]) -> Dict:
    """Detect table headers with score based on boldness, numeric ratio, and top row priority."""
    if not rows:
        return {"headers": [], "body_rows": []}

    scored = []
    row_count = len(rows)
    for idx, row in enumerate(rows):
        if not row:
            scored.append((idx, 0.0))
            continue

        bold_ratio = sum(1 for c in row if c.get("bold")) / len(row)
        numeric_ratio = sum(1 for c in row if _is_numeric(c.get("text", ""))) / len(row)
        top_bonus = 1.0 - (idx / max(1, row_count - 1))
        score = (0.45 * bold_ratio) + (0.35 * (1.0 - numeric_ratio)) + (0.20 * top_bonus)
        scored.append((idx, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_idx, top_score = scored[0]
    header_rows = [top_idx] if top_score >= 0.5 else [0]

    if len(rows) > 1:
        second = 1 if 0 in header_rows else 0
        second_row = rows[second]
        if second_row:
            second_bold = sum(1 for c in second_row if c.get("bold")) / len(second_row)
            second_numeric = sum(1 for c in second_row if _is_numeric(c.get("text", ""))) / len(second_row)
            if second_bold >= 0.25 and second_numeric <= 0.5 and second not in header_rows:
                header_rows.append(second)

    header_rows = sorted(set(header_rows))

    merged_header: List[str] = []
    max_cols = max(len(r) for r in rows)
    for col_idx in range(max_cols):
        parts: List[str] = []
        for r_idx in header_rows:
            if col_idx < len(rows[r_idx]):
                cell_text = rows[r_idx][col_idx].get("text", "").strip()
                if cell_text:
                    parts.append(cell_text)
        merged_header.append(" ".join(parts).strip())

    body_rows = [rows[i] for i in range(len(rows)) if i not in header_rows]
    return {"headers": merged_header, "body_rows": body_rows}
