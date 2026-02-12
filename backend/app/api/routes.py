import os
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Query, UploadFile

from app.api.schemas import ParseResponse
from app.pipeline import parse_document

router = APIRouter()


@router.post("/parse", response_model=ParseResponse)
async def parse_pdf(
    file: UploadFile = File(...),
    debug: bool = Query(default=False),
) -> ParseResponse:
    content = await file.read()
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        parsed = parse_document(tmp_path, debug=debug)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return ParseResponse(**parsed)
