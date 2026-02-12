export interface TableBlock {
  type: "table";
  headers: string[];
  rows: string[][];
}

export interface BulletBlock {
  type: "bullet_list";
  items: string[];
}

export interface ParagraphBlock {
  type: "paragraph";
  content: string;
}

export interface SubSection {
  title: string;
  type: string;
  content?: string;
  items?: string[];
  headers?: string[];
  rows?: string[][];
}

export interface Section {
  title: string;
  subsections: SubSection[];
}

export interface DocumentResponse {
  document_title: string;
  sections: Section[];
}
