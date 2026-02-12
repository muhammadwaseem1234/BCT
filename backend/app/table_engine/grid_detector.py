from typing import Dict, List, Optional

from app.layout.x_clusterer import cluster_x_positions
from app.layout.y_clusterer import cluster_y_positions
from app.table_engine.header_detector import detect_headers


def _nearest_index(values: List[float], target: float) -> int:
    return min(range(len(values)), key=lambda i: abs(values[i] - target))


def _build_grid_from_lines(lines: List[dict], x_tol: float, y_tol: float) -> tuple[List[List[Dict]], List[float], List[float]]:
    col_centers = cluster_x_positions(lines, tolerance=x_tol)
    row_centers = cluster_y_positions(lines, tolerance=y_tol)

    row_map: Dict[int, Dict[int, List[dict]]] = {}
    for line in lines:
        text = line.get("text", "").strip()
        if not text:
            continue
        x0 = float(line["bbox"][0])
        y0 = float(line["bbox"][1])
        r_idx = _nearest_index(row_centers, y0)
        c_idx = _nearest_index(col_centers, x0)
        row_map.setdefault(r_idx, {}).setdefault(c_idx, []).append(line)

    rows: List[List[Dict]] = []
    for r_idx in sorted(row_map.keys()):
        row_cells: List[Dict] = []
        for c_idx in sorted(row_map[r_idx].keys()):
            cell_lines = sorted(row_map[r_idx][c_idx], key=lambda a: a["bbox"][0])
            text = " ".join(c.get("text", "").strip() for c in cell_lines if c.get("text", "").strip()).strip()
            if text:
                row_cells.append(
                    {
                        "text": text,
                        "bold": any(c.get("bold", False) for c in cell_lines),
                        "col": c_idx,
                    }
                )
        if row_cells:
            rows.append(row_cells)

    return rows, col_centers, row_centers


def _render_rows(rows: List[List[Dict]], num_cols: int) -> List[List[str]]:
    rendered_rows: List[List[str]] = []
    for row in rows:
        rendered = [""] * num_cols
        for cell in row:
            rendered[cell["col"]] = cell["text"].strip()
        if any(c for c in rendered):
            rendered_rows.append(rendered)
    return rendered_rows


def _append_cell_text(target: List[str], source: List[str]) -> None:
    for idx, value in enumerate(source):
        if not value:
            continue
        if target[idx]:
            target[idx] = f"{target[idx]} {value}".strip()
        else:
            target[idx] = value


def _merge_wrapped_rows(rows: List[List[str]], description_col: int) -> List[List[str]]:
    if not rows:
        return []

    merged: List[List[str]] = [rows[0][:]]
    for row in rows[1:]:
        non_empty_cols = [i for i, c in enumerate(row) if c.strip()]
        desc_text = row[description_col].strip() if description_col < len(row) else ""
        continuation = False

        if len(non_empty_cols) <= 1:
            continuation = True
        elif desc_text and desc_text[:1].islower():
            continuation = True

        if continuation:
            _append_cell_text(merged[-1], row)
        else:
            merged.append(row[:])

    return merged


def detect_table(lines: List[dict]) -> Optional[dict]:
    """Detect table structure from line bboxes using X/Y clustering and row merging."""
    candidate_lines = [line for line in lines if line.get("text", "").strip()]
    if len(candidate_lines) < 3:
        return None

    rows, col_centers, row_centers = _build_grid_from_lines(candidate_lines, x_tol=24.0, y_tol=4.5)
    if len(rows) < 2 or len(col_centers) < 2:
        return None

    # Keep only grids where at least two columns are consistently populated.
    dense_rows = [row for row in rows if len(row) >= 2]
    if len(dense_rows) < 2:
        return None

    stable_cols = 0
    for c_idx in range(len(col_centers)):
        coverage = sum(1 for row in dense_rows if any(cell["col"] == c_idx for cell in row))
        if coverage >= max(2, int(len(dense_rows) * 0.5)):
            stable_cols += 1

    if stable_cols < 2:
        return None

    header_info = detect_headers(dense_rows)
    headers = header_info["headers"]
    if len(headers) < 2:
        return None

    num_cols = max(len(headers), len(col_centers))
    raw_body_rows = _render_rows(header_info["body_rows"], num_cols)
    description_col = min(len(headers) - 1, num_cols - 1)
    body_rows = _merge_wrapped_rows(raw_body_rows, description_col=description_col)

    if not body_rows:
        return None

    # Require at least one row with multiple populated columns after merge.
    multi_col_rows = [row for row in body_rows if sum(1 for c in row if c.strip()) >= 2]
    if not multi_col_rows:
        return None

    return {
        "headers": headers,
        "rows": body_rows,
        "detected_columns": col_centers,
        "detected_rows": row_centers,
    }
