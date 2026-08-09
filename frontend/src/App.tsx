/**
 * ReGenRAG application shell.
 *
 * Phase 1: session creation, PDF upload, and document listing. Chat, RAG
 * answers, sources, and the self-healing trace belong to later phases.
 */

import SessionPage from "./pages/SessionPage";
import "./styles.css";

export default function App() {
  return (
    <main className="page">
      <header className="header">
        <h1>ReGenRAG</h1>
        <p className="tagline">
          Self-healing retrieval-augmented generation for evidence-grounded
          document intelligence.
        </p>
      </header>
      <SessionPage />
    </main>
  );
}