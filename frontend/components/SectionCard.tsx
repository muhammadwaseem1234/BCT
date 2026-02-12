import { useState } from "react";

import { Section } from "../../shared/types";
import { SubSectionCard } from "./SubSectionCard";

interface Props {
  section: Section;
}

export function SectionCard({ section }: Props) {
  const [open, setOpen] = useState(true);

  return (
    <article className="sectionCard">
      <button className="collapseBtn" onClick={() => setOpen((v) => !v)}>
        {open ? "-" : "+"} {section.title}
      </button>
      {open && (
        <div className="subWrap">
          {section.subsections.map((sub, idx) => (
            <SubSectionCard key={`${sub.title}-${idx}`} subsection={sub} />
          ))}
        </div>
      )}
    </article>
  );
}
