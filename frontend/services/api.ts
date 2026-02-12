import { DocumentResponse } from "../../shared/types";

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export async function parseDocument(file: File, debug = false): Promise<DocumentResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/parse?debug=${debug}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Parse failed: ${response.status}`);
  }

  return (await response.json()) as DocumentResponse;
}
