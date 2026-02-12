from app.table_engine.grid_detector import detect_table


def _line(text: str, y: float, cells: list[tuple[str, float]], bold: bool = False) -> dict:
    atoms = []
    for c_text, x in cells:
        atoms.append({"text": c_text, "bbox": (x, y, x + 40, y + 10), "bold": bold})
    return {"text": text, "atoms": atoms, "bbox": (0, y, 200, y + 10)}


def test_table_detection() -> None:
    lines = [
        _line("Name Score", 10.0, [("Name", 10.0), ("Score", 90.0)], bold=True),
        _line("A 10", 25.0, [("A", 10.0), ("10", 90.0)]),
        _line("B 20", 40.0, [("B", 10.0), ("20", 90.0)]),
    ]

    table = detect_table(lines)
    assert table is not None
    assert len(table["headers"]) >= 2
    assert len(table["rows"]) >= 2
