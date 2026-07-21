"use client";
import { use, useEffect, useState } from "react";
import { api, ConceptRead } from "@/lib/api";
import { X, ChevronLeft, ChevronRight, AlertTriangle } from "lucide-react";
import { useRouter } from "next/navigation";

const LENS_COLORS: Record<string, { bg: string; color: string }> = {
  Literalist:            { bg: "rgba(148,163,184,0.15)", color: "#94a3b8" },
  Contradiction:         { bg: "rgba(249,115,22,0.15)",  color: "#fb923c" },
  Subtraction:           { bg: "rgba(244,63,94,0.13)",   color: "#fb7185" },
  "Wrong Medium":        { bg: "rgba(217,70,239,0.15)",  color: "#e879f9" },
  "Audience Inversion":  { bg: "rgba(20,184,166,0.15)",  color: "#2dd4bf" },
  "Material Honesty":    { bg: "rgba(212,168,83,0.15)",  color: "#d4a853" },
  "Narrative Compression": { bg: "rgba(59,130,246,0.15)", color: "#60a5fa" },
  "Emotional Extremity": { bg: "rgba(239,68,68,0.15)",   color: "#f87171" },
  "Cultural Counter-Signal": { bg: "rgba(16,185,129,0.15)", color: "#34d399" },
  "Constraint Injection":{ bg: "rgba(139,92,246,0.15)",  color: "#a78bfa" },
  "Found Object":        { bg: "rgba(245,158,11,0.15)",  color: "#fbbf24" },
  "Restraint Maximalism":{ bg: "rgba(99,102,241,0.15)",  color: "#818cf8" },
  "Hybrid Concept":      { bg: "rgba(99,102,241,0.10)",  color: "#a5b4fc" },
};

export default function PresentPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);
  const router = useRouter();

  const [concepts, setConcepts] = useState<ConceptRead[]>([]);
  const [current, setCurrent] = useState(0);
  const [loading, setLoading] = useState(true);
  const [showReasoning, setShowReasoning] = useState(false);

  useEffect(() => {
    api.getConcepts(projectId)
      .then(cs => setConcepts(cs.filter(c => c.status === "saved")))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") next();
      if (e.key === "ArrowLeft") prev();
      if (e.key === "Escape") router.push(`/projects/${projectId}/saved`);
      if (e.key === "r") setShowReasoning(v => !v);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  function next() { setCurrent(p => Math.min(p + 1, concepts.length - 1)); setShowReasoning(false); }
  function prev() { setCurrent(p => Math.max(p - 1, 0)); setShowReasoning(false); }

  if (loading) return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", background: "#040508" }}>
      <div className="spinner" />
    </div>
  );

  if (concepts.length === 0) return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", background: "#040508", color: "var(--text-primary)" }}>
      <p style={{ marginBottom: "16px", color: "var(--text-secondary)" }}>No saved concepts to present.</p>
      <button className="btn-secondary" onClick={() => router.push(`/projects/${projectId}/concepts`)}>← Go to Board</button>
    </div>
  );

  const c = concepts[current];
  const ls = LENS_COLORS[c.style_category] ?? LENS_COLORS["Hybrid Concept"];
  const hasDissent = c.critic_notes?.some(n => n.is_dissent);
  const rc = c.reasoning_chain;

  return (
    <div style={{
      minHeight: "100vh",
      background: "#040508",
      display: "grid",
      gridTemplateRows: "1fr auto",
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Ambient glow */}
      <div style={{
        position: "fixed", inset: 0, pointerEvents: "none",
        background: `radial-gradient(ellipse 60% 50% at 50% 20%, ${ls.color}18 0%, transparent 70%)`,
        transition: "background 0.8s ease",
        zIndex: 0,
      }} />

      {/* Close */}
      <button
        onClick={() => router.push(`/projects/${projectId}/saved`)}
        style={{ position: "fixed", top: "20px", right: "24px", zIndex: 10, background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)", borderRadius: "8px", color: "var(--text-secondary)", cursor: "pointer", padding: "8px", display: "flex", alignItems: "center" }}
      >
        <X size={18} />
      </button>

      {/* Slide counter */}
      <div style={{ position: "fixed", top: "20px", left: "50%", transform: "translateX(-50%)", zIndex: 10, fontSize: "12px", color: "var(--text-muted)", fontWeight: 600, letterSpacing: "0.08em" }}>
        {current + 1} / {concepts.length}
      </div>

      {/* Main content */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "80px 48px 40px", position: "relative", zIndex: 1 }}>
        <div style={{ maxWidth: "920px", width: "100%", animation: "fadeIn 0.4s ease" }}>
          <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }`}</style>

          {/* Lens + badges */}
          <div style={{ display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap", marginBottom: "28px" }}>
            <span style={{ padding: "6px 16px", borderRadius: "24px", fontSize: "12px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", background: ls.bg, color: ls.color, border: `1px solid ${ls.color}40` }}>
              {c.style_category}
            </span>
            {hasDissent && <span className="dissent-badge"><AlertTriangle size={10} />Critic Dissent</span>}
            <span className="reference-notice">🔒 Reference only</span>
          </div>

          {/* Title */}
          <h1 style={{ fontSize: "clamp(36px, 5vw, 62px)", fontFamily: "'Playfair Display', serif", fontWeight: 700, lineHeight: 1.12, marginBottom: "24px", letterSpacing: "-0.02em" }}>
            {c.title}
          </h1>

          {/* Core idea */}
          <p style={{ fontSize: "clamp(16px, 2vw, 22px)", color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: "32px", maxWidth: "720px" }}>
            {c.main_visual_idea}
          </p>

          {/* Scores */}
          {c.scores && (
            <div style={{ display: "flex", gap: "12px", marginBottom: "32px", flexWrap: "wrap" }}>
              <ScorePill label="Novelty" value={c.scores.novelty_score} color="#818cf8" />
              <ScorePill label="Feasibility" value={c.scores.feasibility_score} color="#34d399" />
              <ScorePill label="Lens Fidelity" value={c.scores.lens_fidelity_score} color="#d4a853" />
            </div>
          )}

          {/* Toggle reasoning */}
          <button
            onClick={() => setShowReasoning(v => !v)}
            className="btn-ghost"
            style={{ border: "1px solid var(--border)", borderRadius: "8px", padding: "8px 16px", fontSize: "13px", marginBottom: "24px" }}
          >
            {showReasoning ? "Hide" : "Show"} Reasoning Chain <kbd style={{ fontSize: "10px", opacity: 0.5, marginLeft: "4px" }}>R</kbd>
          </button>

          {showReasoning && rc && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "12px", animation: "fadeIn 0.3s ease" }}>
              <ChainCard label="Lens Applied" value={rc.lens_applied} color="#818cf8" />
              <ChainCard label="Tension" value={rc.tension_identified} color="#fb923c" />
              <ChainCard label="Decision" value={rc.design_decision} color="#60a5fa" />
              <ChainCard label="Consequence" value={rc.consequence} color="#34d399" />
              <ChainCard label="Risk" value={rc.risk_flag} color="#f87171" />
            </div>
          )}
        </div>
      </div>

      {/* Navigation */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "16px", padding: "24px", position: "relative", zIndex: 1 }}>
        {/* Dots */}
        <button onClick={prev} disabled={current === 0} style={{ background: "none", border: "none", color: current === 0 ? "var(--text-muted)" : "var(--text-primary)", cursor: current === 0 ? "not-allowed" : "pointer" }}>
          <ChevronLeft size={24} />
        </button>
        <div style={{ display: "flex", gap: "8px" }}>
          {concepts.map((_, i) => (
            <button
              key={i}
              onClick={() => { setCurrent(i); setShowReasoning(false); }}
              style={{ width: i === current ? "24px" : "8px", height: "8px", borderRadius: "4px", background: i === current ? "var(--accent)" : "var(--border)", border: "none", cursor: "pointer", transition: "all 0.2s ease", padding: 0 }}
            />
          ))}
        </div>
        <button onClick={next} disabled={current === concepts.length - 1} style={{ background: "none", border: "none", color: current === concepts.length - 1 ? "var(--text-muted)" : "var(--text-primary)", cursor: current === concepts.length - 1 ? "not-allowed" : "pointer" }}>
          <ChevronRight size={24} />
        </button>
      </div>

      {/* Keyboard hint */}
      <div style={{ position: "fixed", bottom: "12px", right: "20px", fontSize: "11px", color: "var(--text-muted)", zIndex: 10 }}>
        ← → navigate · R reasoning · Esc close
      </div>
    </div>
  );
}

function ScorePill({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ padding: "8px 16px", borderRadius: "10px", background: `${color}12`, border: `1px solid ${color}30`, textAlign: "center" }}>
      <div style={{ fontSize: "20px", fontWeight: 700, color, fontFamily: "monospace" }}>{pct}%</div>
      <div style={{ fontSize: "11px", color: "var(--text-muted)", marginTop: "2px" }}>{label}</div>
    </div>
  );
}

function ChainCard({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{ padding: "14px 16px", borderRadius: "10px", background: `${color}08`, border: `1px solid ${color}20` }}>
      <div style={{ fontSize: "10px", fontWeight: 700, color, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "6px" }}>{label}</div>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.6 }}>{value}</p>
    </div>
  );
}
