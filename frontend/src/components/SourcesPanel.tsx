import { useState } from "react";

interface Source {
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  score: number | null;
}

interface SourcesPanelProps {
  sources: Source[];
}

export default function SourcesPanel({ sources }: SourcesPanelProps) {
  const [expandedSources, setExpandedSources] = useState<Set<string>>(new Set());

  const toggleSource = (sourceId: string) => {
    setExpandedSources((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(sourceId)) {
        newSet.delete(sourceId);
      } else {
        newSet.add(sourceId);
      }
      return newSet;
    });
  };

  if (sources.length === 0) {
    return (
      <section className="card">
        <div className="card-header">
          <h2>Sources</h2>
        </div>
        <p className="muted">No sources found for this answer.</p>
      </section>
    );
  }

  return (
    <section className="card">
      <div className="card-header">
        <h2>Sources ({sources.length})</h2>
      </div>

      <div className="sources-list">
        {sources.map((source) => {
          const sourceKey = `${source.document_id}-${source.chunk_id}`;
          const isExpanded = expandedSources.has(sourceKey);

          return (
            <div
              key={sourceKey}
              className="source-item"
            >
              <div
                className="source-header"
                onClick={() => toggleSource(sourceKey)}
                role="button"
                tabIndex={0}
                aria-expanded={isExpanded}
                aria-label={`Toggle source details for chunk ${source.chunk_index + 1}`}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    toggleSource(sourceKey);
                  }
                }}
              >
                <div className="source-info">
                  <div className="source-meta">
                    <span className="source-id">{source.document_id.slice(0, 8)}…</span>
                    <span className="source-page">
                      Chunk {source.chunk_index + 1}
                      {source.page_number !== null && ` · Page ${source.page_number}`}
                    </span>
                  </div>
                </div>

                <div className="source-score">
                  {source.score !== null ? (
                    <span className={`score-${getScoreClass(source.score)}`}>
                      {source.score.toFixed(2)}
                    </span>
                  ) : (
                    <span className="score-none">N/A</span>
                  )}
                </div>

                <button
                  className="toggle-button"
                  aria-label={isExpanded ? "Collapse source" : "Expand source"}
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleSource(sourceKey);
                  }}
                >
                  {isExpanded ? "−" : "+"}
                </button>
              </div>

              {isExpanded && (
                <div className="source-details">
                  <div className="detail-row">
                    <span className="detail-label">Document ID</span>
                    <span className="detail-value">{source.document_id}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Chunk ID</span>
                    <span className="detail-value">{source.chunk_id}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Chunk Index</span>
                    <span className="detail-value">{source.chunk_index}</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Page Number</span>
                    <span className="detail-value">
                      {source.page_number !== null ? source.page_number : "N/A"}
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Similarity Score</span>
                    <span className="detail-value">
                      {source.score !== null ? source.score.toFixed(3) : "N/A"}
                    </span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function getScoreClass(score: number): string {
  if (score >= 0.8) return "high";
  if (score >= 0.6) return "medium";
  return "low";
}
