/** Types for the RAG query API responses. */

// Mirror of backend RetrievalFailureCategory enum
export type RetrievalFailureCategory =
  | "NO_RESULTS"
  | "LOW_SIMILARITY"
  | "LOW_LLM_RELEVANCE"
  | "INSUFFICIENT_EVIDENCE"
  | "HIGH_RELEVANCE_LOW_SIMILARITY"
  | null;

// Mirror of backend HealingAction enum
export type HealingAction =
  | "QUERY_REWRITE"
  | "RE_RETRIEVE"
  | "ANSWER_REGENERATE"
  | "CONTEXT_IMPROVEMENT"
  | "RETRY_EXHAUSTED";

// Query request
export interface QueryRequest {
  question: string;
}

// Source metadata for a retrieved chunk
export interface QuerySource {
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  page_number: number | null;
  score: number | null;
}

// Retrieval grading information
export interface QueryRetrievalGrading {
  sufficient: boolean;
  threshold: number;
  relevant_count: number;
  low_relevance_count: number;
  best_score: number | null;
  average_score: number | null;
  reason: string;
}

// LLM relevance judgement for a chunk
export interface ChunkRelevanceResponse {
  chunk_id: string;
  relevant: boolean;
  reason: string | null;
  parse_failed: boolean;
}

// Aggregate LLM-based relevance grading
export interface LlmRetrievalGradingResponse {
  relevant_count: number;
  failed_count: number;
  has_relevant_evidence: boolean;
  reason: string;
  judgements: ChunkRelevanceResponse[];
}

// Record of one healing action
export interface HealingStepResponse {
  attempt: number;
  action: HealingAction;
  failure_category: RetrievalFailureCategory;
  details: string;
}

// Main query response
export interface QueryResponse {
  answer: string;
  sources: QuerySource[];
  retrieval_grading: QueryRetrievalGrading | null;
  llm_relevance: LlmRetrievalGradingResponse | null;
  failure_category: RetrievalFailureCategory;
  rewritten_query: string | null;
  healed: boolean;
  attempts: number;
  healing_steps: HealingStepResponse[];
  grounding_score: number | null;
  retry_exhausted: boolean;
}

// Error detail from API
export interface ErrorDetail {
  type: string;
  message: string;
  detail?: string | null;
  request_id?: string | null;
  timestamp: number;
}

// Standardized error response
export interface APIErrorResponse {
  error: ErrorDetail;
}

// System status types
export interface SystemStatus {
  health: {
    status: string;
    service: string;
  };
  ready: {
    status: string;
    service: string;
    database: string;
  };
}
