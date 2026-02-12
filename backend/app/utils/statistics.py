from collections import Counter
from typing import Iterable


def mode(values: Iterable[float], default: float = 0.0) -> float:
    counts = Counter(values)
    if not counts:
        return default
    return counts.most_common(1)[0][0]
