# doc_intelligence

A modular PDF document intelligence stack with FastAPI + Next.js.

## Features

- PDF parsing with PyMuPDF
- H1/H2 section hierarchy detection
- Block classification: paragraph, bullet list, table
- Table extraction using X/Y clustering and header scoring
- FastAPI `/parse` endpoint with optional `debug=true`
- Next.js viewer with collapsible section tree
- Unit tests for section, bullet, and table detection

## Local Run Instructions

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs: `http://localhost:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:3000`

## Parse Endpoint

`POST /parse?debug=true|false`

- Form field: `file` (PDF)

## Example API Response

```json
{
  "document_title": "Quarterly Business Review",
  "sections": [
    {
      "title": "Quarterly Business Review",
      "subsections": [
        {
          "title": "Highlights",
          "type": "bullet_list",
          "items": [
            "Revenue increased by 12%",
            "Churn improved to 3.1%"
          ]
        },
        {
          "title": "Overview",
          "type": "table",
          "headers": ["Region", "Sales"],
          "rows": [["North", "120"], ["South", "98"]]
        }
      ]
    }
  ]
}
```

## Example Frontend Rendering

- Upload panel at top
- Collapsible section cards
- Subsection cards
- Paragraph blocks rendered as text
- Bullet list blocks rendered as `<ul>`
- Table blocks rendered as HTML `<table>`

## Tests

```bash
cd backend
pytest -q
```

## Synthetic Sample PDF

```bash
cd backend
python tests/generate_sample_pdf.py
```

This creates `backend/sample_data/sample.pdf` for manual parsing checks.
