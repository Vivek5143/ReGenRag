import { useState } from "react";

interface AnswerPanelProps {
  answer: string;
}

export default function AnswerPanel({ answer }: AnswerPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const wordCount = answer.trim().split(/\s+/).length;

  return (
    <section className="card">
      <div className="card-header">
        <h2>Answer</h2>
        <span className="muted">{wordCount} words</span>
      </div>

      <div
        className="answer-content"
        style={{
          maxHeight: isExpanded ? "none" : "240px",
          overflow: isExpanded ? "visible" : "auto",
        }}
      >
        <p>{answer}</p>
      </div>

      {wordCount > 50 && (
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="toggle-button"
          aria-label={isExpanded ? "Show less" : "Show more"}
        >
          {isExpanded ? "Show less" : "Show more"}
        </button>
      )}
    </section>
  );
}
