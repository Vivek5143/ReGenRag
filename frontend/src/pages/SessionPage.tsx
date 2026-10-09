import { useEffect, useRef, useState } from "react";
import { createSession, getSession, listDocuments, uploadDocument } from "../services/api";
import type { SessionInfo, UploadedDocument } from "../types/session";

type Phase = "idle" | "creating" | "ready" | "error";

interface SessionPageProps {
  sessionId?: string;
  onSessionCreated?: (id: string) => void;
  onReadyToQuery?: () => void;
  onGoBackToSessionCreation?: () => void;
}

export default function SessionPage({
  sessionId,
  onSessionCreated,
  onReadyToQuery,
  onGoBackToSessionCreation,
}: SessionPageProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const [error, setError] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Fetch session details when sessionId prop is provided
  useEffect(() => {
    if (sessionId) {
      setError("");
      getSession(sessionId)
        .then((fetchedSession) => {
          setSession(fetchedSession);
          setPhase("ready");
          refreshDocuments(sessionId);
        })
        .catch((err) => {
          setError(err instanceof Error ? err.message : "Failed to load session");
          setPhase("error");
        });
    }
  }, [sessionId]);

  const refreshDocuments = async (sessionIdToUse: string) => {
    try {
      const docs = await listDocuments(sessionIdToUse);
      setDocuments(docs);
    } catch (err) {
      console.error("Failed to refresh documents", err);
    }
  };

  // Poll for document processing status
  useEffect(() => {
    const sessionIdToUse = sessionId ?? session?.session_id;
    if (!sessionIdToUse) return;

    const needsPolling = documents.some(
      (doc) => doc.status === "UPLOADED" || doc.status === "PROCESSING"
    );

    if (needsPolling) {
      const interval = setInterval(() => refreshDocuments(sessionIdToUse), 3000);
      return () => clearInterval(interval);
    }
  }, [documents, sessionId, session]);

  const handleStartSession = async () => {
    if (sessionId) return;
    setPhase("creating");
    setError("");
    try {
      const created = await createSession();
      setSession(created);
      setPhase("ready");
      onSessionCreated?.(created.session_id);
    } catch (err) {
      setPhase("error");
      setError(err instanceof Error ? err.message : "Failed to start session");
    }
  };

  const handleUpload = async (file: File | undefined) => {
    const sessionIdToUse = sessionId ?? session?.session_id;
    if (!sessionIdToUse || !file) return;
    setUploading(true);
    setError("");
    try {
      await uploadDocument(sessionIdToUse, file);
      await refreshDocuments(sessionIdToUse);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleCloseSession = async () => {
    if (!sessionId && session) {
      setSession(null);
      setDocuments([]);
      onSessionCreated?.("");
      onGoBackToSessionCreation?.();
    }
  };

  const currentSessionId = sessionId ?? session?.session_id;
  const expiresAt = session?.expires_at;
  const hasProcessedDocs = documents.some((doc) => doc.status === "PROCESSED");

  return (
    <div className="session-page">
      <div className="card">
        <div className="card-header">
          <div>
            <h2>Session</h2>
            <p className="card-subtitle">
              Sessions are transient: uploaded PDFs and their data are removed when the
              session ends or expires.
            </p>
          </div>
          {currentSessionId && (
            <span className={`status-chip status-${(session?.status ?? "").toLowerCase()}`}>
              {session?.status ?? "Unknown"}
            </span>
          )}
        </div>

        {!currentSessionId && (
          <div className="session-empty">
            <p className="muted">Create a new session to start uploading documents.</p>
            <button
              className="btn btn-primary"
              disabled={phase === "creating"}
              onClick={handleStartSession}
            >
              {phase === "creating" ? "Creating session…" : "Create Session"}
            </button>
          </div>
        )}

        {currentSessionId && (
          <div className="session-details">
            <div className="metrics-grid">
              <div className="metric-card">
                <span className="metric-value">{currentSessionId.slice(0, 8)}…</span>
                <span className="metric-label">Session ID</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">{documents.length}</span>
                <span className="metric-label">Documents</span>
              </div>
              <div className="metric-card">
                <span className="metric-value">
                  {expiresAt ? new Date(expiresAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : "—"}
                </span>
                <span className="metric-label">Expires</span>
              </div>
            </div>

            <div className="upload-section">
              <h3>Upload Document</h3>
              <div className="upload-row">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  disabled={uploading}
                  onChange={(e) => void handleUpload(e.target.files?.[0])}
                  aria-label="Upload PDF document"
                />
                {uploading && <span className="muted">Uploading…</span>}
                {!sessionId && (
                  <button
                    className="btn btn-secondary"
                    onClick={handleCloseSession}
                  >
                    End Session
                  </button>
                )}
              </div>
            </div>

            {documents.length > 0 && (
              <div className="documents-section">
                <h3>Documents</h3>
                <ul className="doc-list">
                  {documents.map((doc) => (
                    <li key={doc.document_id} className="doc-item">
                      <div className="doc-info">
                        <span className="doc-name">{doc.filename}</span>
                        <span className="doc-meta">
                          {(doc.file_size / 1024).toFixed(1)} KB
                        </span>
                      </div>
                      <span className={`status-chip status-${doc.status.toLowerCase()}`}>
                        {doc.status}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {hasProcessedDocs && onReadyToQuery && (
              <div className="ready-to-query">
                <button
                  className="btn btn-primary"
                  onClick={onReadyToQuery}
                >
                  Start Querying →
                </button>
                <p className="muted">
                  You have processed documents. Click above to start asking questions.
                </p>
              </div>
            )}
          </div>
        )}

        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
