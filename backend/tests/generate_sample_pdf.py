"""Generate a small synthetic PDF for manual parsing checks."""

from pathlib import Path

import fitz


def generate(path: str) -> None:
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((72, 72), "Quarterly Business Review", fontsize=22, fontname="helv")
    page.insert_text((72, 110), "Highlights", fontsize=16, fontname="helv")
    page.insert_text((86, 135), "- Revenue increased by 12%", fontsize=11, fontname="helv")
    page.insert_text((86, 150), "- Churn improved to 3.1%", fontsize=11, fontname="helv")

    page.insert_text((72, 190), "Region", fontsize=12, fontname="helv")
    page.insert_text((180, 190), "Sales", fontsize=12, fontname="helv")
    page.insert_text((72, 210), "North", fontsize=11, fontname="helv")
    page.insert_text((180, 210), "120", fontsize=11, fontname="helv")
    page.insert_text((72, 228), "South", fontsize=11, fontname="helv")
    page.insert_text((180, 228), "98", fontsize=11, fontname="helv")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    doc.close()


if __name__ == "__main__":
    generate("sample_data/sample.pdf")
