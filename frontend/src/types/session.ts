/** Shared API types for the session/document Phase 1 UI. */

export type SessionStatus = "ACTIVE" | "EXPIRED" | "CLOSED";
export type DocumentStatus =
  | "UPLOADED"
  | "PROCESSING"
  | "PROCESSED"
  | "FAILED"
  | "DELETED";

export interface SessionInfo {
  session_id: string;
  status: SessionStatus;
  created_at?: string;
  last_activity?: string;
  expires_at: string;
}

export interface UploadedDocument {
  document_id: string;
  session_id: string;
  filename: string;
  status: DocumentStatus;
  file_size: number;
  created_at: string;
  processed_at?: string | null;
}

export interface HealthStatus {
  status: string;
  service: string;
  database: string;
}