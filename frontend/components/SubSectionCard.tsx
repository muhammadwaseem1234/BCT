import { useState } from "react";

import { SubSection } from "../../shared/types";
import { BulletListBlock } from "./BulletListBlock";
import { ParagraphBlock } from "./ParagraphBlock";
import { TableBlock } from "./TableBlock";

interface Props {
  subsection: SubSection;
}

export function SubSectionCard({ subsection }: Props) {
  const [open, setOpen] = useState(true);

  return (
    <section className="subSectionCard">
      <button className="collapseBtn small" onClick={() => setOpen((v) => !v)}>
        {open ? "-" : "+"} {subsection.title}
      </button>
      {open && subsection.type === "paragraph" && (
        <ParagraphBlock content={subsection.content ?? ""} />
      )}
      {open && subsection.type === "bullet_list" && (
        <BulletListBlock items={subsection.items ?? []} />
      )}
      {open && subsection.type === "table" && (
        <TableBlock headers={subsection.headers ?? []} rows={subsection.rows ?? []} />
      )}
    </section>
  );
}
