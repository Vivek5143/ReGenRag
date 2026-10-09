import { useState } from "react";

interface RetrievalAnalysisProps {
  response: {
    answer: string;
    sources: Array<{
      document_id: string;
      chunk_id: string;
      chunk_index: number;
      page_number: number | null;
      score: number | null;
    }>;
    retrieval_grading: {
      sufficient: boolean;
      threshold: number;
      relevant_count: number;
      low_relevance_count: number;
      best_score: number | null;
      average_score: number | null;
      reason: string;
    } | null;
    llm_relevance: {
      relevant_count: number;
      failed_count: number;
      has_relevant_evidence: boolean;
      reason: string;
      judgements: Array<{
        chunk_id: string;
        relevant: boolean;
        reason: string | null;
        parse_failed: boolean;
      }>;
    } | null;
    failure_category:
      | "NO_RESULTS"
      | "LOW_SIMILARITY"
      | "LOW_LLM_RELEVANCE"
      | "INSUFFICIENT_EVIDENCE"
      | "HIGH_RELEVANCE_LOW_SIMILARITY"
      | null;
    rewritten_query: string | null;
    healed: boolean;
    attempts: number;
    healing_steps: Array<{
      attempt: number;
      action:
        | "QUERY_REWRITE"
        | "RE_RETRIEVE"
        | "ANSWER_REGENERATE"
        | "CONTEXT_IMPROVEMENT"
        | "RETRY_EXHAUSTED";
      failure_category:
        | "NO_RESULTS"
        | "LOW_SIMILARITY"
        | "LOW_LLM_RELEVANCE"
        | "INSUFFICIENT_EVIDENCE"
        | "HIGH_RELEVANCE_LOW_SIMILARITY"
        | null;
      details: string;
    }>;
    grounding_score: number | null;
    retry_exhausted: boolean;
  };
}

export default function RetrievalAnalysis({ response }: RetrievalAnalysisProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const getFailureCategoryText = (category: RetrievalAnalysisProps["response"]["failure_category"]): string => {
    switch (category) {
      case "NO_RESULTS": return "No results retrieved";
      case "LOW_SIMILARITY": return "Low similarity scores";
      case "LOW_LLM_RELEVANCE": return "Low LLM relevance";
      case "INSUFFICIENT_EVIDENCE": return "Insufficient evidence";
      case "HIGH_RELEVANCE_LOW_SIMILARITY": return "LLM-relevant but low similarity";
      default: return "";
    }
  };

  const getHealingActionText = (action: NonNullable<NonNullable<RetrievalAnalysisProps["response"]["healing_steps"]>[number]["action"]>): string => {
    switch (action) {
      case "QUERY_REWRITE": return "Query rewritten";
      case "RE_RETRIEVE": return "Re-retrieved";
      case "ANSWER_REGENERATE": return "Answer regenerated";
      case "CONTEXT_IMPROVEMENT": return "Context improved";
      case "RETRY_EXHAUSTED": return "Retries exhausted";
      default: return action;
    }
  };

  if (!response) {
    return null;
  }

  const {
    retrieval_grading,
    llm_relevance,
    failure_category,
    rewritten_query,
    healed,
    attempts,
    healing_steps,
    grounding_score,
    retry_exhausted,
  } = response;

  const initialRetrievalSufficient = retrieval_grading?.sufficient ?? false;
  const hasHealingSteps = healing_steps.length > 0;
  const selfHealingOccurred = healed && !retry_exhausted && attempts > 1;

  return (
    <section className="card">
      <div className="card-header">
        <h2>Retrieval Analysis</h2>
        <span className={`status-badge ${initialRetrievalSufficient ? "success" : "warning"}`}>
          {initialRetrievalSufficient ? "✓ Sufficient" : "⚠ Insufficient"}
        </span>
      </div>

      {/* Pipeline Visualization */}
      <div className="pipeline">
        <div className={`pipeline-step ${initialRetrievalSufficient ? "success" : "warning"}`}>
          <div className="pipeline-icon">1</div>
          <div className="pipeline-label">Retrieve</div>
          <div className="pipeline-status">
            {retrieval_grading?.relevant_count ?? 0} chunks
          </div>
        </div>

        <div className={`pipeline-step ${retrieval_grading?.sufficient ? "success" : "warning"}`}>
          <div className="pipeline-icon">2</div>
          <div className="pipeline-label">Grade</div>
          <div className="pipeline-status">
            {retrieval_grading?.sufficient ? "Pass" : "Fail"}
          </div>
        </div>

        {llm_relevance && (
          <div className={`pipeline-step ${llm_relevance.has_relevant_evidence ? "success" : "skipped"}`}>
            <div className="pipeline-icon">3</div>
            <div className="pipeline-label">LLM</div>
            <div className="pipeline-status">
              {llm_relevance.relevant_count}/{llm_relevance.relevant_count + llm_relevance.failed_count}
            </div>
          </div>
        )}

        <div className={`pipeline-step ${!retry_exhausted && response.answer !== "I couldn't find enough relevant information" ? "success" : "error"}`}>
          <div className="pipeline-icon">4</div>
          <div className="pipeline-label">Generate</div>
          <div className="pipeline-status">
            {retry_exhausted ? "Failed" : "Done"}
          </div>
        </div>

        <div className={`pipeline-step ${grounding_score !== null ? (grounding_score! >= 0.7 ? "success" : "warning") : "skipped"}`}>
          <div className="pipeline-icon">5</div>
          <div className="pipeline-label">Ground</div>
          <div className="pipeline-status">
            {grounding_score !== null ? grounding_score!.toFixed(2) : "—"}
          </div>
        </div>
      </div>

      {/* Retrieval Metrics */}
      {retrieval_grading && (
        <div className="metrics-grid">
          <div className="metric-card">
            <span className="metric-value">{retrieval_grading.threshold}</span>
            <span className="metric-label">Threshold</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{retrieval_grading.relevant_count}</span>
            <span className="metric-label">Relevant</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{retrieval_grading.low_relevance_count}</span>
            <span className="metric-label">Low Rel.</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">{retrieval_grading.best_score?.toFixed(2) ?? "—"}</span>
            <span className="metric-label">Best Score</span>
          </div>
        </div>
      )}

      {/* Failure Category */}
      {!initialRetrievalSufficient && failure_category && (
        <div className="alert alert-warning">
          <strong>Retrieval insufficient:</strong> {getFailureCategoryText(failure_category)}
          {retrieval_grading?.reason && (
            <p className="muted" style={{ marginTop: "0.5rem" }}>{retrieval_grading.reason}</p>
          )}
        </div>
      )}

      {/* Healing Steps */}
      {hasHealingSteps && (
        <div className="healing-steps">
          <h3>Healing Process</h3>

          {selfHealingOccurred && (
            <div className="status-badge success" style={{ marginBottom: "1rem" }}>
              ✓ Successfully recovered after {attempts} attempt{attempts > 1 ? "s" : ""}
            </div>
          )}

          {retry_exhausted && (
            <div className="status-badge error" style={{ marginBottom: "1rem" }}>
              ✗ Recovery failed — retries exhausted
            </div>
          )}

          {rewritten_query && (
            <div className="query-update">
              <strong>Query rewritten:</strong>
              <code>{rewritten_query}</code>
            </div>
          )}

          <ol className="healing-steps-list">
            {healing_steps.map((step, index) => (
              <li key={index} className="healing-step">
                <span className="step-number">{index + 1}</span>
                <div className="step-content">
                  <span className="step-action">{getHealingActionText(step.action)}</span>
                  {step.failure_category && (
                    <span className="step-failure">
                      ({getFailureCategoryText(step.failure_category)})
                    </span>
                  )}
                  {step.details && (
                    <div className="step-details">{step.details}</div>
                  )}
                </div>
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Grounding Score */}
      {grounding_score !== null && (
        <div className="metric-card" style={{ marginTop: "1rem" }}>
          <span className={`metric-value ${grounding_score >= 0.7 ? "" : "text-warning"}`}>
            {grounding_score.toFixed(2)}
          </span>
          <span className="metric-label">
            Grounding Score {grounding_score >= 0.7 ? "✓" : "⚠ (threshold: 0.7)"}
          </span>
        </div>
      )}

      {/* LLM Relevance */}
      {llm_relevance && (
        <div className="alert alert-info" style={{ marginTop: "1rem" }}>
          <strong>LLM Relevance:</strong> {llm_relevance.reason}
        </div>
      )}

      {/* Toggle Details */}
      <div
        className="analysis-footer"
        onClick={() => setIsExpanded(!isExpanded)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setIsExpanded(!isExpanded);
          }
        }}
      >
        <span>{isExpanded ? "Hide details" : "Show details"}</span>
        <span className="toggle-button">{isExpanded ? "−" : "+"}</span>
      </div>

      {isExpanded && (
        <div className="collapsed-notice">
          <p>Detailed retrieval analysis including all grading metrics and healing steps.</p>
        </div>
      )}
    </section>
  );
}
