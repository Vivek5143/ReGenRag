/**
 * Phase 1 session page: create a temporary session, upload PDFs, and list the
 * uploaded documents. The chat/RAG UI belongs to later phases.
 */

import { useRef, useState } from "react";

import { createSession, listDocuments, uploadDocument } from "../services/api";
import type { SessionInfo, UploadedDocument } from "../types/session";

type Phase = "idle" | "creating" | "ready" | "error";

export default function SessionPage() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [error, setError] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refreshDocuments = async (sessionId: string) => {
    setDocuments(await listDocuments(sessionId));
  };

  const handleStartSession = async () => {
    setPhase("creating");
    setError("");
    try {
      const created = await createSession();
      setSession(created);
      setPhase("ready");
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : "Failed to start session");
    }
  };

  const handleUpload = async (file: File | undefined) => {
    if (!session || !file) return;
    setUploading(true);
    setError("");
    try {
      await uploadDocument(session.session_id, file);
      await refreshDocuments(session.session_id);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <section className="card">
      <h2>Temporary Session</h2>

      <p className="muted">
        Sessions are transient: uploaded PDFs and their data are removed when the
        session ends or expires.
      </p>

      {!session && (
        <button disabled={phase === "creating"} onClick={handleStartSession}>
          {phase === "creating" ? "Starting…" : "Start Session"}
        </button>
      )}

      {session && (
        <>
          <dl className="session-meta">
            <div>
              <dt>Session ID</dt>
              <dd>{session.session_id}</dd>
            </div>
            <div>
              <dt>Status</dt>
              <dd>{session.status}</dd>
            </div>
            <div>
              <dt>Expires</dt>
              <dd>{new Date(session.expires_at).toLocaleString()}</dd>
            </div>
          </dl>

          <div className="upload-row">
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf,.pdf"
              disabled={uploading}
              onChange={(e) => void handleUpload(e.target.files?.[0])}
            />
            {uploading && <span className="muted">Uploading…</span>}
          </div>

          <h3>Uploaded documents</h3>
          {documents.length === 0 ? (
            <p className="muted">No documents uploaded yet.</p>
          ) : (
            <ul className="doc-list">
              {documents.map((doc) => (
                <li key={doc.document_id}>
                  <span className="doc-name">{doc.filename}</span>
                  <span className="status-chip">{doc.status}</span>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      {error && <p className="error">{error}</p>}
    </section>
  );
}