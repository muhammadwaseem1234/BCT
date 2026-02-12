import { useState } from "react";

import { ChunkTreeView } from "../components/ChunkTreeView";
import { parseDocument } from "../services/api";
import { DocumentResponse } from "../../shared/types";

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const [doc, setDoc] = useState<DocumentResponse | null>(null);

  const onSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError("Select a PDF file first.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const parsed = await parseDocument(file);
      setDoc(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1 className="title">Document Intelligence Viewer</h1>
      <form className="uploadPanel" onSubmit={onSubmit}>
        <input
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Parsing..." : "Upload and Parse"}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {doc && <ChunkTreeView document={doc} />}
    </main>
  );
}
