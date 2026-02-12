interface Props {
  content: string;
}

export function ParagraphBlock({ content }: Props) {
  return <p className="paragraph">{content}</p>;
}
