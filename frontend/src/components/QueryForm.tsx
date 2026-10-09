import { useState } from "react";

interface QueryFormProps {
  onSubmit: (question: string) => Promise<void>;
  onError: (error: unknown) => void;
}

export default function QueryForm({
  onSubmit,
  onError,
}: QueryFormProps) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;

    setLoading(true);
    setError(null);
    try {
      await onSubmit(question);
      setQuestion("");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      onError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="query-form">
      <div className="form-group">
        <label htmlFor="question-input" className="form-label">
          Ask your documents
        </label>
        <textarea
          id="question-input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={4}
          placeholder="What would you like to know about your documents?"
          disabled={loading}
          aria-label="Enter your question"
        />
      </div>

      {error && (
        <p className="error">{error}</p>
      )}

      <div className="submit-row">
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="btn btn-primary"
        >
          {loading ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Processing...
            </>
          ) : (
            "Ask Question"
          )}
        </button>
      </div>
    </form>
  );
}
