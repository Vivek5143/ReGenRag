import { useState } from "react";
import SessionPage from "./pages/SessionPage";
import QueryPage from "./pages/QueryPage";
import "./styles.css";

type Page = "session" | "query";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("session");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isReadyToQuery, setIsReadyToQuery] = useState(false);

  const handleSessionCreated = (id: string) => {
    setSessionId(id);
    setIsReadyToQuery(false);
    setCurrentPage("session");
  };

  const handleReadyToQuery = () => {
    setIsReadyToQuery(true);
    setCurrentPage("query");
  };

  const handleBackToSession = () => {
    setSessionId(null);
    setIsReadyToQuery(false);
    setCurrentPage("session");
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">R</div>
            <div>
              <h1>ReGenRAG</h1>
              <p className="tagline">
                Self-healing retrieval-augmented generation for evidence-grounded
                document intelligence.
              </p>
            </div>
          </div>

          {sessionId && (
            <nav className="nav-tabs">
              <button
                className={`nav-tab ${currentPage === "session" ? "active" : ""}`}
                onClick={() => setCurrentPage("session")}
              >
                Session
              </button>
              <button
                className={`nav-tab ${currentPage === "query" ? "active" : ""}`}
                onClick={() => setCurrentPage("query")}
                disabled={!isReadyToQuery}
              >
                Query
              </button>
            </nav>
          )}
        </div>
      </header>

      <main className="main-content">
        {currentPage === "session" ? (
          <SessionPage
            sessionId={sessionId || undefined}
            onSessionCreated={handleSessionCreated}
            onReadyToQuery={handleReadyToQuery}
            onGoBackToSessionCreation={handleBackToSession}
          />
        ) : (
          <QueryPage sessionId={sessionId!} />
        )}
      </main>
    </div>
  );
}
