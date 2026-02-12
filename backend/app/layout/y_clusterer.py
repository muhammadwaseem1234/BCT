from typing import Iterable, List


def cluster_y_positions(blocks: Iterable[dict], tolerance: float = 3.0) -> List[float]:
    """Cluster y positions and return representative centroids."""
    rows: List[float] = []
    counts: List[int] = []

    for block in blocks:
        y0 = float(block["bbox"][1])
        matched = False
        for i, row in enumerate(rows):
            if abs(row - y0) <= tolerance:
                counts[i] += 1
                rows[i] = rows[i] + (y0 - rows[i]) / counts[i]
                matched = True
                break
        if not matched:
            rows.append(y0)
            counts.append(1)

    return sorted(rows)
