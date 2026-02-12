interface Props {
  items: string[];
}

export function BulletListBlock({ items }: Props) {
  return (
    <ul className="bulletList">
      {items.map((item, idx) => (
        <li key={`${item}-${idx}`}>{item}</li>
      ))}
    </ul>
  );
}
