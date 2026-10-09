import { useState, useEffect } from "react";
import QueryForm from "../components/QueryForm";
import AnswerPanel from "../components/AnswerPanel";
import SourcesPanel from "../components/SourcesPanel";
import RetrievalAnalysis from "../components/RetrievalAnalysis";
import SystemStatus from "../components/SystemStatus";
import { querySession } from "../services/api";
import type { QueryResponse } from "../types/query";

interface QueryPageProps {
  sessionId: string;
}

export default function QueryPage({ sessionId }: QueryPageProps) {
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);

  // Fetch system status periodically
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await (window as any).fetchSystemStatus();
        setSystemStatus(status);
      } catch (err) {
        // Silently fail for status checks
      }
    };

    const interval = setInterval(fetchStatus, 5000);
    fetchStatus();
    return () => clearInterval(interval);
  }, []);

  const handleSubmit = async (question: string) => {
    try {
      const result = await querySession(sessionId, { question });
      setResponse(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      setResponse(null);
    }
  };

  return (
    <div className="query-page">
      <div className="page-header">
        <h1>Ask Your Documents</h1>
        <p className="subtitle">
          ReGenRAG evaluates evidence and automatically recovers when initial
          retrieval is insufficient.
        </p>
      </div>

      <SystemStatus status={systemStatus} />

      <div className="query-layout">
        <div className="query-input-section">
          <QueryForm
            onSubmit={handleSubmit}
            onError={(err) => setError(err instanceof Error ? err.message : "Unknown error")}
          />

          {error && (
            <div className="alert alert-error">
              {error}
            </div>
          )}
        </div>

        {response && (
          <div className="query-results">
            <AnswerPanel answer={response.answer} />
            <SourcesPanel sources={response.sources} />
            <RetrievalAnalysis response={response} />
          </div>
        )}
      </div>
    </div>
  );
}
