import { DocumentResponse } from "../../shared/types";
import { SectionCard } from "./SectionCard";

interface Props {
  document: DocumentResponse;
}

export function ChunkTreeView({ document }: Props) {
  return (
    <section className="treeWrap">
      <h2>{document.document_title}</h2>
      {document.sections.map((section) => (
        <SectionCard key={section.title} section={section} />
      ))}
    </section>
  );
}
