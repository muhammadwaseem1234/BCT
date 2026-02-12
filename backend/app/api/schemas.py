from typing import List, Literal, Optional

from pydantic import BaseModel


class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[str]]


class BulletBlock(BaseModel):
    type: Literal["bullet_list"] = "bullet_list"
    items: List[str]


class ParagraphBlock(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    content: str


class SubSection(BaseModel):
    title: str
    type: Literal["paragraph", "bullet_list", "table"]
    content: Optional[str] = None
    items: Optional[List[str]] = None
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None


class Section(BaseModel):
    title: str
    subsections: List[SubSection]


class DocumentResponse(BaseModel):
    document_title: str
    sections: List[Section]


class ParseDebug(BaseModel):
    raw_blocks: List[dict]
    detected_columns: List[float]
    detected_rows: List[float]


class ParseResponse(BaseModel):
    document_title: str
    sections: List[Section]
    debug: Optional[ParseDebug] = None
