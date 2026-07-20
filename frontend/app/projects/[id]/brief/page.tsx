"use client";
import { use, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { api, CreativeBrief, ClarifyingQuestion, Project } from "@/lib/api";
import { ArrowRight, Zap, AlertTriangle, ChevronDown, ChevronUp, Check, SkipForward } from "lucide-react";

export default function BriefPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);
  const router = useRouter();

  const [project, setProject] = useState<Project | null>(null);
  const [brief, setBrief] = useState<CreativeBrief | null>(null);
  const [questions, setQuestions] = useState<ClarifyingQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [loadingBrief, setLoadingBrief] = useState(true);
  const [loadingQuestions, setLoadingQuestions] = useState(false);
  const [generatingConcepts, setGeneratingConcepts] = useState(false);
  const [questionsExpanded, setQuestionsExpanded] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const [p, b] = await Promise.all([api.getProject(projectId), api.getBrief(projectId).catch(() => null)]);
        setProject(p);
        if (b) {
          setBrief(b);
          const qs = await api.getQuestions(projectId);
          setQuestions(qs);
          // Pre-fill existing answers
          const existingAnswers: Record<number, string> = {};
          qs.forEach((q) => { if (q.answer) existingAnswers[q.id] = q.answer; });
          setAnswers(existingAnswers);
        } else {
          // Auto-extract brief
          const extracted = await api.extractBrief(projectId);
          setBrief(extracted);
        }
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        setLoadingBrief(false);
      }
    })();
  }, [projectId]);

  async function handleGenerateQuestions() {
    setLoadingQuestions(true);
    try {
      const qs = await api.generateQuestions(projectId);
      setQuestions(qs);
      setAnswers({});
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate questions");
    } finally {
      setLoadingQuestions(false);
    }
  }

  async function handleAnswerQuestion(qId: number, skip = false) {
    try {
      await api.answerQuestion(qId, {
        answer: skip ? undefined : answers[qId],
        skipped: skip,
      });
    } catch {}
  }

  async function handleGenerateConcepts() {
    setGeneratingConcepts(true);
    // Submit all answers
    for (const q of questions) {
      if (answers[q.id] || q.skipped) await handleAnswerQuestion(q.id, !answers[q.id]);
    }
    try {
      await api.generateConcepts(projectId);
      router.push(`/projects/${projectId}/concepts`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Concept generation failed");
      setGeneratingConcepts(false);
    }
  }

  if (loadingBrief) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Extracting creative brief…</p>
    </div>
  );

  if (error && !brief) return (
    <div className="page-shell">
      <div className="glass-card" style={{ padding: "32px", textAlign: "center" }}>
        <p style={{ color: "var(--danger)", marginBottom: "12px" }}>{error}</p>
        <Link href="/dashboard" className="btn-secondary">← Dashboard</Link>
      </div>
    </div>
  );

  return (
    <div className="page-shell" style={{ maxWidth: "800px" }}>
      {/* Breadcrumb */}
      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "32px", fontSize: "13px", color: "var(--text-muted)" }}>
        <Link href="/dashboard" style={{ color: "var(--text-muted)", textDecoration: "none" }}>Dashboard</Link>
        <span>/</span>
        <span style={{ color: "var(--text-secondary)" }}>{project?.title}</span>
        <span>/</span>
        <span style={{ color: "var(--text-primary)" }}>Brief</span>
      </div>

      <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", marginBottom: "6px" }}>Creative Brief</h1>
      <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "32px" }}>
        Extracted from your imagination. Review, then generate clarifying questions or proceed to concepts.
      </p>

      {brief && (
        <div className="glass-card" style={{ padding: "28px", marginBottom: "24px" }}>
          {/* Brief fields */}
          <BriefRow label="Main Subject" value={brief.main_subject} />
          <BriefRow label="Design Type" value={brief.design_type} />
          <BriefRow label="Target Audience" value={brief.target_audience} />
          <BriefRow label="Mood" value={brief.mood.join(" · ")} />
          <BriefRow label="Color Palette" value={brief.colors.join(", ")} />
          <BriefRow label="Fixed Elements" value={brief.fixed_elements.join(", ")} />
          <BriefRow label="Avoid" value={brief.avoid_elements.join(", ")} />

          {/* Tensions — V2 Contradiction Engine output */}
          {brief.tensions && brief.tensions.length > 0 && (
            <div style={{ marginTop: "24px", padding: "16px", borderRadius: "12px", background: "rgba(249,115,22,0.07)", border: "1px solid rgba(249,115,22,0.2)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
                <AlertTriangle size={14} style={{ color: "var(--dissent)" }} />
                <span style={{ fontSize: "12px", fontWeight: 700, color: "var(--dissent)", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Contradiction Engine — Tensions Identified
                </span>
              </div>
              {brief.tensions.map((t, i) => (
                <p key={i} style={{ fontSize: "13px", color: "#fdba74", lineHeight: 1.65, marginBottom: i < brief.tensions.length - 1 ? "8px" : 0 }}>
                  {t}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Clarifying Questions */}
      <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: questions.length ? "20px" : "0" }}>
          <div>
            <h2 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "4px" }}>Clarifying Questions</h2>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>Answer to sharpen the concepts — all optional</p>
          </div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            {questions.length > 0 && (
              <button className="btn-ghost" onClick={() => setQuestionsExpanded(!questionsExpanded)}>
                {questionsExpanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>
            )}
            <button
              className="btn-secondary"
              style={{ fontSize: "13px", padding: "8px 16px" }}
              onClick={handleGenerateQuestions}
              disabled={loadingQuestions}
            >
              {loadingQuestions ? <><span className="spinner" />Generating…</> : questions.length ? "Regenerate" : "Generate Questions"}
            </button>
          </div>
        </div>

        {questionsExpanded && questions.map((q, i) => (
          <div key={q.id} style={{ padding: "16px 0", borderTop: "1px solid var(--border)" }}>
            <p style={{ fontSize: "13px", fontWeight: 500, marginBottom: "10px" }}>
              <span style={{ color: "var(--text-muted)", marginRight: "8px" }}>Q{i + 1}</span>
              {q.question}
            </p>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                className="input-field"
                placeholder="Your answer (optional)…"
                value={answers[q.id] ?? ""}
                onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
                style={{ flex: 1 }}
              />
              <button className="btn-success" onClick={() => handleAnswerQuestion(q.id)}>
                <Check size={13} />
              </button>
              <button className="btn-ghost" onClick={() => handleAnswerQuestion(q.id, true)}>
                <SkipForward size={13} />
              </button>
            </div>
          </div>
        ))}
      </div>

      {error && (
        <div style={{ padding: "12px 16px", borderRadius: "10px", background: "rgba(244,63,94,0.08)", border: "1px solid rgba(244,63,94,0.2)", color: "var(--danger)", fontSize: "13px", marginBottom: "20px" }}>
          {error}
        </div>
      )}

      <button
        className="btn-primary"
        style={{ width: "100%", justifyContent: "center", padding: "14px" }}
        onClick={handleGenerateConcepts}
        disabled={generatingConcepts}
      >
        {generatingConcepts
          ? <><span className="spinner" />Generating 10 reasoning directions…</>
          : <>Generate 10 Concept Directions <ArrowRight size={16} /></>}
      </button>
    </div>
  );
}

function BriefRow({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "12px", padding: "10px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", paddingTop: "2px" }}>{label}</span>
      <span style={{ fontSize: "14px", color: "var(--text-secondary)" }}>{value}</span>
    </div>
  );
}
