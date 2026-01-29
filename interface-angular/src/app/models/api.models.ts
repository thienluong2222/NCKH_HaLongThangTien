// ==========================================
// API Response Models - ICHeritage Backend
// ==========================================

// Health Check
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  timestamp: string;
  services: {
    yolo_pipeline: boolean;
    bayesian_classifier: boolean;
    gemini_api_key: boolean;
  };
  history_count: number;
}

// Festivals List
export interface FestivalInfo {
  name: string;
  total_constraints: number;
  hard_constraints: number;
  soft_constraints: number;
}

export interface FestivalsResponse {
  total: number;
  festivals: FestivalInfo[];
}

// Media Analysis (Video/Image)
export interface ConstraintResult {
  type: string;
  params: string[];
  is_hard: boolean;
  weight: number;
  threshold?: number;
  satisfied: boolean;
}

export interface FestivalConstraints {
  festival: string;
  confidence: number;
  satisfied: ConstraintResult[];
  unsatisfied: ConstraintResult[];
}

export interface Top3Festival {
  festival: string;
  confidence: number;
}

// Analysis - needs clarification (works for both video and image)
export interface AnalysisNeedsClarificationResponse {
  status: 'needs_clarification';
  request_id: string;
  history_id: string;
  question: string;
  target_features: string[];
  candidates_preliminary: string[];
  top_3?: Top3Festival[];  // Optional - may not be present
  top_3_constraints: FestivalConstraints[];
  detected_objects_count: number;
}

// Analysis - finished (works for both video and image)
export interface AnalysisFinishedResponse {
  status: 'finished';
  history_id: string;
  result: string[] | null;  // Can be null if no festival detected
  message?: string;  // Message when result is null
  probabilities?: Record<string, number>;
  top_3?: Top3Festival[];  // Optional - may not be present
  top_3_constraints: FestivalConstraints[];
  detected_objects_count: number;
}

export type AnalysisResponse = AnalysisNeedsClarificationResponse | AnalysisFinishedResponse;

// Legacy type aliases for backward compatibility
export type VideoAnalysisResponse = AnalysisResponse;
export type VideoNeedsClarificationResponse = AnalysisNeedsClarificationResponse;
export type VideoFinishedResponse = AnalysisFinishedResponse;

// Answer submission
export interface AnswerRequest {
  request_id: string;
  answer: string;
}

export interface AnswerResponse {
  status: 'needs_clarification' | 'finished';
  history_id: string;
  request_id?: string; // Only if needs more clarification
  question?: string;
  target_features?: string[];
  result?: string[];
  probabilities?: Record<string, number>;
  top_3?: Top3Festival[];  // Optional - may not be present
  top_3_constraints: FestivalConstraints[];
  analysis_breakdown?: Record<string, boolean>;
}

// History
export interface HistoryItem {
  id: string;
  filename: string;
  status: 'processing' | 'needs_clarification' | 'finished' | 'error';
  winner: string | null;
  created_at: string;
}

export interface HistoryListResponse {
  total: number;
  limit: number;
  offset: number;
  data: HistoryItem[];
}

export interface QAItem {
  question: string;
  answer: string;
  timestamp: string;
}

export interface HistoryDetailResponse {
  id: string;
  filename: string;
  status: string;
  detected_objects: any[];
  qa_history: QAItem[];
  result: {
    winner: string;
    top_3: Top3Festival[];
  } | null;
  created_at: string;
  updated_at: string;
}

// Error Response
export interface ApiError {
  error: string;
  code: string;
  details?: any;
}
