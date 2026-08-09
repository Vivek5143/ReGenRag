/**
 * Thin API client for the ReGenRAG backend.
 *
 * All calls go through the Vite dev proxy (/api -> http://localhost:8000), so
 * the frontend makes no direct external requests in development.
 */

import type { HealthStatus, SessionInfo, UploadedDocument } from "../types/session";

const BASE = "/api/v1";

async function handle<T>(response: Response): Promise<T> {
  if (response.status === 204) {
    return undefined as T;
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body; keep the status text
      detail = `${detail}: ${response.statusText}`;
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export async function createSession(): Promise<SessionInfo> {
  return handle(
    await fetch(`${BASE}/sessions`, { method: "POST" }),
  );
}

export async function getSession(sessionId: string): Promise<SessionInfo> {
  return handle(await fetch(`${BASE}/sessions/${sessionId}`));
}

export async function closeSession(sessionId: string): Promise<void> {
  return handle(await fetch(`${BASE}/sessions/${sessionId}`, { method: "DELETE" }));
}

export async function listDocuments(sessionId: string): Promise<UploadedDocument[]> {
  return handle(await fetch(`${BASE}/sessions/${sessionId}/documents`));
}

export async function uploadDocument(
  sessionId: string,
  file: File,
): Promise<UploadedDocument> {
  const form = new FormData();
  form.append("file", file);
  return handle(
    await fetch(`${BASE}/sessions/${sessionId}/documents`, {
      method: "POST",
      body: form,
    }),
  );
}

export async function fetchHealth(): Promise<HealthStatus> {
  return handle(await fetch("/health"));
}