"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead, Project } from "@/lib/api";
import {
  Bookmark, BookmarkCheck, X, RefreshCw, Combine,
  AlertTriangle, Eye, ChevronDown, ChevronUp, Sliders,
  Anchor, GitCompare, Minus, Tv, History, Users, Hammer, Film,
  Cpu, Flame, AlertCircle, Lock, Eye as EyeIcon, Smile,
  Compass, Rocket, Volume2, Maximize, Scale, Scroll, HelpCircle
} from "lucide-react";

// Accent colors and icons map per lens
const LENS_METADATA: Record<string, { color: string; bg: string; icon: React.ReactNode }> = {
  Literalist: {
    color: "#94a3b8",
    bg: "rgba(148,163,184,0.12)",
    icon: <Anchor size={14} />,
  },
  Contradiction: {
    color: "#fb923c",
    bg: "rgba(249,115,22,0.12)",
    icon: <GitCompare size={14} />,
  },
  Subtraction: {
    color: "#fb7185",
    bg: "rgba(244,63,94,0.10)",
    icon: <Minus size={14} />,
  },
  "Wrong Medium": {
    color: "#e879f9",
    bg: "rgba(217,70,239,0.12)",
    icon: <Tv size={14} />,
  },
  "Historical Transplant": {
    color: "#60a5fa",
    bg: "rgba(59,130,246,0.12)",
    icon: <History size={14} />,
  },
  "Audience Inversion": {
    color: "#2dd4bf",
    bg: "rgba(20,184,166,0.12)",
    icon: <Users size={14} />,
  },
  "Material Honesty": {
    color: "#fbbf24",
    bg: "rgba(245,158,11,0.12)",
    icon: <Hammer size={14} />,
  },
  "Narrative Compression": {
    color: "#60a5fa",
    bg: "rgba(59,130,246,0.12)",
    icon: <Film size={14} />,
  },
  "Systems View": {
    color: "#22d3ee",
    bg: "rgba(34,211,238,0.12)",
    icon: <Cpu size={14} />,
  },
  "Emotional Extremity": {
    color: "#f87171",
    bg: "rgba(239,68,68,0.12)",
    icon: <Flame size={14} />,
  },
  "Cultural Counter-Signal": {
    color: "#34d399",
    bg: "rgba(16,185,129,0.12)",
    icon: <AlertCircle size={14} />,
  },
  "Constraint Injection": {
    color: "#a78bfa",
    bg: "rgba(139,92,246,0.12)",
    icon: <Lock size={14} />,
  },
  "Found Object": {
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.12)",
    icon: <EyeIcon size={14} />,
  },
  "Absurdist Logic": {
    color: "#fbbf24",
    bg: "rgba(251,191,36,0.12)",
    icon: <Smile size={14} />,
  },
  "Restraint Maximalism": {
    color: "#818cf8",
    bg: "rgba(99,102,241,0.12)",
    icon: <Compass size={14} />,
  },
  "Temporal Displacement": {
    color: "#c084fc",
    bg: "rgba(192,132,252,0.12)",
    icon: <Rocket size={14} />,
  },
  "Sensory Translation": {
    color: "#f97316",
    bg: "rgba(249,115,22,0.12)",
    icon: <Volume2 size={14} />,
  },
  "Negative Space Narrative": {
    color: "#14b8a6",
    bg: "rgba(20,184,166,0.12)",
    icon: <Maximize size={14} />,
  },
  "Scale Distortion": {
    color: "#fbbf24",
    bg: "rgba(251,191,36,0.12)",
    icon: <Scale size={14} />,
  },
  "Provenance Fiction": {
    color: "#a855f7",
    bg: "rgba(168,85,247,0.12)",
    icon: <Scroll size={14} />,
  },
  "Hybrid Concept": {
    color: "#a5b4fc",
    bg: "rgba(99,102,241,0.08)",
    icon: <GitCompare size={14} />,
  },
};

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
  const [generating, setGenerating] = useState(false);
  const [creativeDistance, setCreativeDistance] = useState<number>(3); // Slider (1 to 5)

  useEffect(() => {
    (async () => {
      try {
        const [p, cs] = await Promise.all([api.getProject(projectId), api.getConcepts(projectId)]);
        setProject(p);
        setConcepts(cs);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Failed to load concepts");
      } finally {
        setLoading(false);
      }
    })();
  }, [projectId]);

  async function handleFullGenerate() {
    setGenerating(true);
    setError("");
    try {
      const cs = await api.generateConcepts(projectId, creativeDistance);
      setConcepts(cs);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate concepts");
    } finally {
      setGenerating(false);
    }
  }

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
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px", flexWrap: "wrap", gap: "24px" }}>
        <div>
          <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "8px", fontSize: "13px", color: "var(--text-muted)" }}>
            <Link href="/dashboard" style={{ color: "var(--text-muted)", textDecoration: "none" }}>Dashboard</Link>
            <span>/</span>
            <span>{project?.title}</span>
            <span>/</span>
            <span style={{ color: "var(--text-primary)" }}>Concepts</span>
          </div>
          <h1 style={{ fontSize: "32px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Concept Board
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
            {active.length} directions · {saved.length} saved
            {combineMode ? <span style={{ marginLeft: "12px", color: "#fbbf24" }}>· Combine mode: select a second concept</span> : ""}
          </p>
        </div>

        {/* Creative Distance Slider */}
        <div className="glass-card" style={{ padding: "16px 20px", display: "flex", alignItems: "center", gap: "16px", minWidth: "300px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "4px", flex: 1 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", fontWeight: 600 }}>
              <span style={{ color: "var(--text-muted)" }}>Creative Distance</span>
              <span style={{ color: "#a5b4fc" }}>
                {creativeDistance === 1 ? "Safe (Literal)" : creativeDistance === 5 ? "Adventurous (High Risk)" : `Level ${creativeDistance}`}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="5"
              step="1"
              value={creativeDistance}
              onChange={(e) => setCreativeDistance(parseInt(e.target.value))}
              style={{ width: "100%", accentColor: "var(--accent)", cursor: "pointer" }}
            />
          </div>
          <button
            onClick={handleFullGenerate}
            disabled={generating}
            className="btn-primary"
            style={{ padding: "10px 16px", fontSize: "13px", whiteSpace: "nowrap" }}
          >
            {generating ? <span className="spinner" style={{ borderLeftColor: "#fff" }} /> : <><RefreshCw size={12} /> Regenerate All</>}
          </button>
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
  const [scoresExpanded, setScoresExpanded] = useState(false);
  const lensMeta = LENS_METADATA[c.style_category] ?? LENS_METADATA["Hybrid Concept"];
  const hasDissent = c.critic_notes?.some((n) => n.is_dissent);
  const dissentNotes = c.critic_notes?.filter((n) => n.is_dissent) ?? [];
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
      {/* Accent band */}
      <div style={{ height: "5px", background: `linear-gradient(90deg, ${lensMeta.color}, ${lensMeta.color}33)` }} />

      <div style={{ padding: "20px" }}>
        {/* Top tag row */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "14px" }}>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", alignItems: "center" }}>
            <span className="lens-tag" style={{ background: lensMeta.bg, color: lensMeta.color, display: "inline-flex", alignItems: "center", gap: "4px" }}>
              {lensMeta.icon}
              {c.style_category}
            </span>
            {hasDissent && <span className="dissent-badge" style={{ background: "rgba(249,115,22,0.12)", color: "#fb923c", border: "1px solid rgba(249,115,22,0.2)" }}><AlertTriangle size={10} />Dissent</span>}
            {isSaved && <span style={{ padding: "3px 9px", borderRadius: "6px", background: "rgba(52,211,153,0.1)", color: "var(--success)", border: "1px solid rgba(52,211,153,0.2)", fontSize: "11px", fontWeight: 600 }}>Saved</span>}
          </div>
          <span style={{ fontSize: "11px", color: "var(--text-muted)", fontWeight: 600 }}>
            #{c.concept_number}
          </span>
        </div>

        {/* Title & visual idea */}
        <h3 style={{ fontSize: "17px", fontWeight: 700, marginBottom: "8px", lineHeight: 1.35 }}>{c.title}</h3>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: "16px" }}>
          {c.main_visual_idea}
        </p>

        {/* Expandable score chips */}
        {c.scores && (
          <div style={{ display: "flex", flexDirection: "column", gap: "6px", marginBottom: "16px" }}>
            <div 
              onClick={() => setScoresExpanded(!scoresExpanded)} 
              style={{ display: "flex", gap: "6px", flexWrap: "wrap", cursor: "pointer" }}
              title="Click to view details"
            >
              <span className="score-chip novelty">Novelty: {c.scores.novelty_score.toFixed(1)}/5</span>
              <span className="score-chip feasibility">Feasibility: {c.scores.feasibility_score.toFixed(1)}/5</span>
              <span className="score-chip fidelity">Fidelity: {c.scores.lens_fidelity_score.toFixed(1)}/5</span>
            </div>
            {scoresExpanded && (
              <div style={{ fontSize: "11px", color: "var(--text-muted)", padding: "10px", background: "rgba(255,255,255,0.02)", borderRadius: "8px", border: "1px solid var(--border)", display: "flex", flexDirection: "column", gap: "6px" }}>
                <div><strong>Novelty:</strong> Concept uniqueness vs history and trends.</div>
                <div><strong>Feasibility:</strong> {c.scores.feasibility_reason || "Implementation complexity and producer review."}</div>
                <div><strong>Lens Fidelity:</strong> How strictly this concept embodies the {c.style_category} moves.</div>
              </div>
            )}
          </div>
        )}

        {/* First-class Dissent alerts */}
        {hasDissent && (
          <div style={{ marginBottom: "16px", padding: "10px 12px", borderRadius: "8px", background: "rgba(249,115,22,0.08)", border: "1px solid rgba(249,115,22,0.25)", display: "flex", flexDirection: "column", gap: "4px" }}>
            <span style={{ fontSize: "11px", fontWeight: 700, color: "#fb923c", display: "flex", alignItems: "center", gap: "4px" }}>
              <AlertTriangle size={12} /> CRITIC PANEL DISSENT
            </span>
            {dissentNotes.map((note, idx) => (
              <p key={idx} style={{ fontSize: "12px", color: "#fdba74", lineHeight: 1.5, margin: 0 }}>
                "{note.note_text}"
              </p>
            ))}
          </div>
        )}

        {/* Watermark / Reference notice */}
        <span className="reference-notice" style={{ marginBottom: "16px", display: "inline-flex" }}>
          🔒 Reference only — final artwork belongs to the designer.
        </span>

        {/* Quick actions row */}
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

        {/* Concept management controls */}
        {!isCombinable && !isCombineSource && (
          <div style={{ borderTop: "1px solid var(--border)", paddingTop: "14px", marginTop: "14px", display: "flex", flexDirection: "column", gap: "10px" }}>
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
            <p style={{ fontSize: "12px", color: "#a5b4fc", marginBottom: "8px" }}>Select another concept card to blend with this direction.</p>
            <button className="btn-ghost" onClick={onCancelCombine} style={{ fontSize: "12px" }}>Cancel combine</button>
          </div>
        )}
      </div>
    </div>
  );
}
