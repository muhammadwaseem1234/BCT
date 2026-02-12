from typing import Iterable, List


def cluster_x_positions(blocks: Iterable[dict], tolerance: float = 10.0) -> List[float]:
    """Cluster x positions and return representative left-edge centroids."""
    columns: List[float] = []
    counts: List[int] = []

    for block in blocks:
        x0 = float(block["bbox"][0])
        matched = False
        for i, col in enumerate(columns):
            if abs(col - x0) <= tolerance:
                counts[i] += 1
                columns[i] = columns[i] + (x0 - columns[i]) / counts[i]
                matched = True
                break
        if not matched:
            columns.append(x0)
            counts.append(1)

    return sorted(columns)
