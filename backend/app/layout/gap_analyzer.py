def has_large_vertical_gap(prev_block: dict, current_block: dict, threshold: float = 20.0) -> bool:
    """Return True when vertical whitespace between two blocks is large."""
    return abs(float(current_block["bbox"][1]) - float(prev_block["bbox"][3])) > threshold
