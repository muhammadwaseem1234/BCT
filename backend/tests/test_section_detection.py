from app.hierarchy.section_detector import detect_sections


def test_section_and_subsection_detection() -> None:
    blocks = [
        {"text": "Annual Report 2025", "font_size": 20.0, "bold_ratio": 1.0, "bbox": (0, 0, 100, 20), "page": 0},
        {"text": "Overview", "font_size": 16.0, "bold_ratio": 1.0, "bbox": (0, 40, 90, 55), "page": 0},
        {"text": "Revenue increased by 20%.", "font_size": 11.0, "bold_ratio": 0.0, "bbox": (0, 70, 240, 85), "page": 0},
    ]

    sections, _ = detect_sections(blocks)

    assert sections[0]["title"] == "Annual Report 2025"
    assert sections[0]["subsections"][1]["title"] == "Overview"
