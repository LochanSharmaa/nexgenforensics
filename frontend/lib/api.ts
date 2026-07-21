// Frontend API client — all calls go through FastAPI backend only
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail ?? "API error");
  }
  return res.json() as Promise<T>;
}

// ── Projects ────────────────────────────────────────────────
export const api = {
  health: () => apiFetch<{ status: string; app: string }>("/health"),

  createProject: (body: { title: string; raw_imagination: string; design_type?: string }) =>
    apiFetch<Project>("/projects", { method: "POST", body: JSON.stringify(body) }),

  getProjects: () => apiFetch<Project[]>("/projects"),

  getProject: (id: number) => apiFetch<Project>(`/projects/${id}`),

  // ── Brief ────────────────────────────────────────────────
  extractBrief: (project_id: number) =>
    apiFetch<CreativeBrief>("/brief/extract", {
      method: "POST",
      body: JSON.stringify({ project_id }),
    }),

  getBrief: (project_id: number) => apiFetch<CreativeBrief>(`/brief/${project_id}`),

  generateQuestions: (project_id: number) =>
    apiFetch<ClarifyingQuestion[]>("/brief/questions", {
      method: "POST",
      body: JSON.stringify({ project_id }),
    }),

  getQuestions: (project_id: number) =>
    apiFetch<ClarifyingQuestion[]>(`/projects/${project_id}/questions`),

  answerQuestion: (question_id: number, body: { answer?: string; skipped: boolean }) =>
    apiFetch<ClarifyingQuestion>(`/brief/questions/${question_id}/answer`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // ── Concepts ─────────────────────────────────────────────
  generateConcepts: (project_id: number, creative_distance?: number) =>
    apiFetch<ConceptRead[]>("/concepts/generate", {
      method: "POST",
      body: JSON.stringify({ project_id, creative_distance }),
    }),

  getConcepts: (project_id: number) =>
    apiFetch<ConceptRead[]>(`/projects/${project_id}/concepts`),

  saveConcept: (project_id: number, concept_id: number) =>
    apiFetch<ConceptRead>("/concepts/save", {
      method: "POST",
      body: JSON.stringify({ project_id, concept_id, instruction: "save" }),
    }),

  rejectConcept: (project_id: number, concept_id: number) =>
    apiFetch<ConceptRead>("/concepts/reject", {
      method: "POST",
      body: JSON.stringify({ project_id, concept_id, instruction: "reject" }),
    }),

  regenerateConcept: (project_id: number, concept_id: number, instruction: string) =>
    apiFetch<ConceptRead>("/concepts/regenerate", {
      method: "POST",
      body: JSON.stringify({ project_id, concept_id, instruction }),
    }),

  combineConcepts: (project_id: number, concept_a_id: number, concept_b_id: number, custom_instruction?: string) =>
    apiFetch<ConceptRead>("/concepts/combine", {
      method: "POST",
      body: JSON.stringify({ project_id, concept_a_id, concept_b_id, custom_instruction }),
    }),

  checkDiversity: (project_id: number) =>
    apiFetch<DiversityCheckResult>("/concepts/diversity-check", {
      method: "POST",
      body: JSON.stringify({ project_id }),
    }),

  // ── Exports ───────────────────────────────────────────────
  exportJson: (project_id: number) =>
    fetch(`${API_BASE}/export/${project_id}/json`).then((r) => r.blob()),

  exportMarkdown: (project_id: number) =>
    fetch(`${API_BASE}/export/${project_id}/markdown`).then((r) => r.blob()),

  // ── Settings ──────────────────────────────────────────────
  getProviderSettings: () => apiFetch<ProviderSettings>("/settings/providers"),

  testLLM: (provider: string) =>
    apiFetch<TestResult>("/settings/test-llm", {
      method: "POST",
      body: JSON.stringify({ provider }),
    }),

  testImage: (provider: string) =>
    apiFetch<TestResult>("/settings/test-image", {
      method: "POST",
      body: JSON.stringify({ provider }),
    }),

  // ── Images ────────────────────────────────────────────
  generateImage: (concept_id: number) =>
    apiFetch<ImageJobRead>("/images/generate", {
      method: "POST",
      body: JSON.stringify({ concept_id }),
    }),

  generateImageBatch: (project_id: number, concept_ids: number[]) =>
    apiFetch<ImageJobRead[]>("/images/generate-batch", {
      method: "POST",
      body: JSON.stringify({ project_id, concept_ids }),
    }),

  getImageJob: (job_id: string) =>
    apiFetch<ImageJobRead>(`/images/jobs/${job_id}`),

  getProjectImages: (project_id: number) =>
    apiFetch<ImageResponse[]>(`/projects/${project_id}/images`),

  // ── Missing Opportunities ─────────────────────────────
  getMissingOpportunities: (project_id: number) =>
    apiFetch<MissingOpportunityReport>(`/projects/${project_id}/missing-opportunities`).catch(() =>
      // Fallback: derive from concepts if endpoint not available
      apiFetch<ConceptRead[]>(`/projects/${project_id}/concepts`).then(cs => ({
        report: "Missing Opportunities analysis requires concept generation first.",
        gaps: [],
      }))
    ),

  // ── Genealogy ─────────────────────────────────────────
  getGenealogy: (project_id: number) =>
    apiFetch<GenealogyNode[]>(`/projects/${project_id}/genealogy`).catch(() =>
      apiFetch<ConceptRead[]>(`/projects/${project_id}/concepts`).then(cs =>
        cs.map(c => ({
          id: c.id,
          title: c.title,
          lens: c.style_category,
          parent_id: null,
          generation: 1,
          status: c.status,
        }))
      )
    ),
};

// ── Types (mirroring backend Pydantic schemas) ────────────────
export interface Project {
  id: number;
  title: string;
  raw_imagination: string;
  design_type?: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface CreativeBrief {
  id: number;
  project_id: number;
  main_subject: string;
  design_type: string;
  target_audience: string;
  mood: string[];
  colors: string[];
  fixed_elements: string[];
  flexible_elements: string[];
  avoid_elements: string[];
  tensions: string[];
  created_at: string;
}

export interface ClarifyingQuestion {
  id: number;
  project_id: number;
  question: string;
  answer?: string;
  skipped: boolean;
  created_at: string;
}

export interface ReasoningChain {
  lens_applied: string;
  tension_identified: string;
  design_decision: string;
  consequence: string;
  risk_flag: string;
}

export interface ConceptScore {
  novelty_score: number;
  feasibility_score: number;
  feasibility_reason?: string;
  lens_fidelity_score: number;
}

export interface CriticNote {
  agent_role: string;
  note_text: string;
  is_dissent: boolean;
}

export interface ConceptRead {
  id: number;
  project_id: number;
  concept_number: number;
  title: string;
  style_category: string;
  main_visual_idea: string;
  composition: string;
  lighting: string;
  background: string;
  color_palette: string[];
  typography_direction: string;
  creative_twist: string;
  designer_execution_notes: string;
  reference_image_prompt: string;
  reference_only_notice: string;
  metaphor: string;
  rejected_metaphor_1: string;
  rejected_metaphor_2: string;
  camera_language: string;
  lighting_reasoning: string;
  material_reasoning: string;
  anti_pattern: string;
  status: string;
  diversity_score?: number;
  created_at: string;
  reasoning_chain?: ReasoningChain;
  scores?: ConceptScore;
  critic_notes: CriticNote[];
  parent_id?: number | null;
  generation?: number;
}

export interface DiversityCheckResult {
  overall_diversity_score: number;
  pairwise_warnings: {
    concept_a: number;
    concept_b: number;
    similarity: number;
    issue: string;
    regeneration_instruction: string;
  }[];
}

export interface ProviderSettings {
  llm_provider: string;
  llm_model: string;
  image_provider: string;
  embedding_provider: string;
  available_llm_providers: string[];
  available_image_providers: string[];
  available_embedding_providers: string[];
}

export interface TestResult {
  success: boolean;
  latency_ms: number;
  message: string;
}

export interface ImageJobRead {
  id: string;
  concept_id: number;
  status: "queued" | "running" | "completed" | "failed";
  image_url?: string;
  error_message_sanitized?: string;
  created_at: string;
}

export interface ImageResponse {
  image_url: string;
  prompt: string;
  provider: string;
  reference_only: boolean;
  notice: string;
}

export interface MissingOpportunityReport {
  report: string;
  gaps: string[];
}

export interface GenealogyNode {
  id: number;
  title: string;
  lens: string;
  parent_id: number | null;
  generation: number;
  status: string;
}
