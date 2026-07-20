"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead, Project } from "@/lib/api";
import {
  Bookmark, BookmarkCheck, X, RefreshCw, Combine,
  AlertTriangle, Eye, ChevronDown, ChevronUp
} from "lucide-react";

const LENS_COLORS: Record<string, { bg: string; color: string }> = {
  Literalist:            { bg: "rgba(148,163,184,0.12)", color: "#94a3b8" },
  Contradiction:         { bg: "rgba(249,115,22,0.12)",  color: "#fb923c" },
  Subtraction:           { bg: "rgba(244,63,94,0.10)",   color: "#fb7185" },
  "Wrong Medium":        { bg: "rgba(217,70,239,0.12)",  color: "#e879f9" },
  "Audience Inversion":  { bg: "rgba(20,184,166,0.12)",  color: "#2dd4bf" },
  "Material Honesty":    { bg: "rgba(212,168,83,0.12)",  color: "#d4a853" },
  "Narrative Compression":{ bg: "rgba(59,130,246,0.12)", color: "#60a5fa" },
  "Emotional Extremity": { bg: "rgba(239,68,68,0.12)",   color: "#f87171" },
  "Cultural Counter-Signal": { bg: "rgba(16,185,129,0.12)", color: "#34d399" },
  "Constraint Injection":{ bg: "rgba(139,92,246,0.12)",  color: "#a78bfa" },
  "Found Object":        { bg: "rgba(245,158,11,0.12)",  color: "#fbbf24" },
  "Restraint Maximalism":{ bg: "rgba(99,102,241,0.12)",  color: "#818cf8" },
  "Hybrid Concept":      { bg: "rgba(99,102,241,0.08)",  color: "#a5b4fc" },
};

function scorePct(v: number) { return `${Math.round(v * 100)}%`; }

export default function ConceptsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);

  const [project, setProject] = useState<Project | null>(null);
  const [concepts, setConcepts] = useState<ConceptRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [regeneratingId, setRegeneratingId] = useState<number | null>(null);
  const [combineMode, setCombineMode] = useState<number | null>(null);
  const [regenInstruction, setRegenInstruction] = useState<Record<number, string>>({});
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const [p, cs] = await Promise.all([api.getProject(projectId), api.getConcepts(projectId)]);
        setProject(p);
        setConcepts(cs);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load");
      } finally {
        setLoading(false);
      }
    })();
  }, [projectId]);

  async function handleSave(c: ConceptRead) {
    const updated = await api.saveConcept(projectId, c.id);
    setConcepts((prev) => prev.map((x) => (x.id === c.id ? updated : x)));
  }
  async function handleReject(c: ConceptRead) {
    const updated = await api.rejectConcept(projectId, c.id);
    setConcepts((prev) => prev.map((x) => (x.id === c.id ? updated : x)));
  }
  async function handleRegenerate(c: ConceptRead) {
    const instruction = regenInstruction[c.id] || "Provide a more distinct and creative direction.";
    setRegeneratingId(c.id);
    try {
      const updated = await api.regenerateConcept(projectId, c.id, instruction);
      setConcepts((prev) => prev.map((x) => (x.id === c.id ? updated : x)));
    } finally {
      setRegeneratingId(null);
    }
  }
  async function handleCombine(targetId: number) {
    if (!combineMode) return;
    try {
      const combined = await api.combineConcepts(projectId, combineMode, targetId);
      setConcepts((prev) => [...prev, combined]);
      setCombineMode(null);
    } catch {}
  }

  const QUICK_REGEN = [
    { label: "More luxury", value: "Make this more luxury, premium, and refined" },
    { label: "More minimal", value: "Simplify to extreme minimalism" },
    { label: "More cinematic", value: "Add cinematic drama, wider framing, bolder lighting" },
    { label: "Higher risk", value: "Push this to a genuinely uncomfortable, high-risk creative direction" },
  ];

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Loading concept board…</p>
    </div>
  );

  const saved = concepts.filter((c) => c.status === "saved");
  const active = concepts.filter((c) => c.status !== "rejected");

  return (
    <div className="page-shell">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "8px", fontSize: "13px", color: "var(--text-muted)" }}>
            <Link href="/dashboard" style={{ color: "var(--text-muted)", textDecoration: "none" }}>Dashboard</Link>
            <span>/</span>
            <span>{project?.title}</span>
            <span>/</span>
            <span style={{ color: "var(--text-primary)" }}>Concepts</span>
          </div>
          <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Concept Board
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
            {active.length} directions · {saved.length} saved
            {combineMode ? <span style={{ marginLeft: "12px", color: "#fbbf24" }}>· Combine mode: select a second concept</span> : ""}
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <Link href={`/projects/${projectId}/saved`} className="btn-secondary" style={{ fontSize: "13px" }}>
            <Bookmark size={14} /> Saved ({saved.length})
          </Link>
          <Link href={`/projects/${projectId}/brief`} className="btn-ghost" style={{ fontSize: "13px" }}>
            ← Brief
          </Link>
        </div>
      </div>

      {error && (
        <div style={{ marginBottom: "24px", padding: "12px 16px", borderRadius: "10px", background: "rgba(244,63,94,0.08)", border: "1px solid rgba(244,63,94,0.2)", color: "var(--danger)", fontSize: "13px" }}>
          {error}
        </div>
      )}

      {/* Grid */}
      <div className="concept-grid">
        {active.map((c) => (
          <ConceptCard
            key={c.id}
            concept={c}
            projectId={projectId}
            combineMode={combineMode}
            regenInstruction={regenInstruction[c.id] || ""}
            onInstructionChange={(v) => setRegenInstruction((prev) => ({ ...prev, [c.id]: v }))}
            onSave={() => handleSave(c)}
            onReject={() => handleReject(c)}
            onRegenerate={() => handleRegenerate(c)}
            onCombineSelect={() => combineMode ? handleCombine(c.id) : setCombineMode(c.id)}
            onCancelCombine={() => setCombineMode(null)}
            isRegenerating={regeneratingId === c.id}
            isCombinable={combineMode !== null && combineMode !== c.id}
            isCombineSource={combineMode === c.id}
            quickRegen={QUICK_REGEN}
          />
        ))}
      </div>
    </div>
  );
}

interface CardProps {
  concept: ConceptRead;
  projectId: number;
  combineMode: number | null;
  regenInstruction: string;
  onInstructionChange: (v: string) => void;
  onSave: () => void;
  onReject: () => void;
  onRegenerate: () => void;
  onCombineSelect: () => void;
  onCancelCombine: () => void;
  isRegenerating: boolean;
  isCombinable: boolean;
  isCombineSource: boolean;
  quickRegen: { label: string; value: string }[];
}

function ConceptCard({
  concept: c, projectId, regenInstruction, onInstructionChange,
  onSave, onReject, onRegenerate, onCombineSelect, onCancelCombine,
  isRegenerating, isCombinable, isCombineSource, quickRegen
}: CardProps) {
  const [expanded, setExpanded] = useState(false);
  const lensStyle = LENS_COLORS[c.style_category] ?? LENS_COLORS["Hybrid Concept"];
  const hasDissent = c.critic_notes?.some((n) => n.is_dissent);
  const isSaved = c.status === "saved";

  return (
    <div
      className="glass-card"
      style={{
        padding: "0",
        overflow: "hidden",
        outline: isCombineSource ? "2px solid var(--accent)" : isCombinable ? "1px dashed var(--accent)" : "none",
        cursor: isCombinable ? "pointer" : "default",
      }}
      onClick={isCombinable ? onCombineSelect : undefined}
    >
      {/* Color band */}
      <div style={{ height: "4px", background: `linear-gradient(90deg, ${lensStyle.color}80, ${lensStyle.color}20)` }} />

      <div style={{ padding: "20px" }}>
        {/* Top row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "14px" }}>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            <span className="lens-tag" style={{ background: lensStyle.bg, color: lensStyle.color }}>
              {c.style_category}
            </span>
            {hasDissent && <span className="dissent-badge"><AlertTriangle size={10} />Dissent</span>}
            {isSaved && <span style={{ ...Object.assign({} as React.CSSProperties), padding: "3px 9px", borderRadius: "6px", background: "rgba(52,211,153,0.1)", color: "var(--success)", border: "1px solid rgba(52,211,153,0.2)", fontSize: "11px", fontWeight: 600 }}>Saved</span>}
          </div>
          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 600 }}>
            #{c.concept_number}
          </span>
        </div>

        {/* Title + idea */}
        <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "8px", lineHeight: 1.35 }}>{c.title}</h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: "16px" }}>
          {c.main_visual_idea}
        </p>

        {/* Scores */}
        {c.scores && (
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "16px" }}>
            <span className="score-chip novelty">N {scorePct(c.scores.novelty_score)}</span>
            <span className="score-chip feasibility">F {scorePct(c.scores.feasibility_score)}</span>
            <span className="score-chip fidelity">L {scorePct(c.scores.lens_fidelity_score)}</span>
          </div>
        )}

        {/* Reference notice */}
        <span className="reference-notice" style={{ marginBottom: "16px", display: "inline-flex" }}>
          🔒 Reference only
        </span>

        {/* Expandable reasoning view */}
        {!isCombinable && (
          <button
            className="btn-ghost"
            style={{ width: "100%", justifyContent: "center", marginBottom: "4px" }}
            onClick={() => setExpanded(!expanded)}
          >
            <Eye size={13} /> {expanded ? "Hide" : "View"} Reasoning Chain
            {expanded ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
          </button>
        )}

        {expanded && (
          <div style={{ marginTop: "12px" }}>
            <Link
              href={`/projects/${projectId}/concepts/${c.id}`}
              className="btn-secondary"
              style={{ width: "100%", justifyContent: "center", fontSize: "13px" }}
            >
              Full Reasoning View →
            </Link>
          </div>
        )}

        {/* Actions */}
        {!isCombinable && !isCombineSource && (
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "14px", marginTop: "14px", display: "flex", flexDirection: "column", gap: "10px" }}>
            {/* Quick regen buttons */}
            <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
              {quickRegen.map((q) => (
                <button
                  key={q.label}
                  className="btn-ghost"
                  style={{ fontSize: "11px", padding: "4px 10px", border: "1px solid var(--border)", borderRadius: "6px" }}
                  onClick={() => { onInstructionChange(q.value); onRegenerate(); }}
                >
                  {q.label}
                </button>
              ))}
            </div>

            {/* Custom regen */}
            <div style={{ display: "flex", gap: "6px" }}>
              <input
                className="input-field"
                style={{ fontSize: "12px", padding: "7px 12px" }}
                placeholder="Custom instruction…"
                value={regenInstruction}
                onChange={(e) => onInstructionChange(e.target.value)}
              />
              <button className="btn-ghost" style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "7px 12px", whiteSpace: "nowrap" }} onClick={onRegenerate} disabled={isRegenerating}>
                {isRegenerating ? <span className="spinner" /> : <RefreshCw size={13} />}
              </button>
            </div>

            {/* Save / Reject / Combine */}
            <div style={{ display: "flex", gap: "6px" }}>
              <button className="btn-success" onClick={onSave} style={{ flex: 1, justifyContent: "center" }}>
                {isSaved ? <><BookmarkCheck size={13} />Saved</> : <><Bookmark size={13} />Save</>}
              </button>
              <button className="btn-danger" onClick={onReject} style={{ flex: 1, justifyContent: "center" }}>
                <X size={13} />Reject
              </button>
              <button className="btn-ghost" onClick={onCombineSelect} style={{ border: "1px solid var(--border)", borderRadius: "8px", flex: 1, justifyContent: "center" }}>
                <Combine size={13} />Combine
              </button>
            </div>
          </div>
        )}

        {isCombineSource && (
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "14px", marginTop: "14px" }}>
            <p style={{ fontSize: "12px", color: "#a5b4fc", marginBottom: "8px" }}>Now select another concept to combine with this one.</p>
            <button className="btn-ghost" onClick={onCancelCombine} style={{ fontSize: "12px" }}>Cancel combine</button>
          </div>
        )}
      </div>
    </div>
  );
}
