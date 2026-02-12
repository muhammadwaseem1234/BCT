from app.classifiers.block_classifier import classify_subsection


def test_bullet_detection() -> None:
    subsection = {
        "title": "Highlights",
        "lines": [
            {"text": "- Revenue up", "atoms": [], "bbox": (0, 0, 100, 10)},
            {"text": "- Margin improved", "atoms": [], "bbox": (0, 12, 120, 22)},
            {"text": "- Debt reduced", "atoms": [], "bbox": (0, 24, 100, 34)},
        ],
    }

    parsed, _ = classify_subsection(subsection)
    assert parsed["type"] == "bullet_list"
    assert parsed["items"] == ["Revenue up", "Margin improved", "Debt reduced"]
