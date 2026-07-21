"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead } from "@/lib/api";
import { BookmarkCheck, ArrowLeft, Download, FileText, Presentation } from "lucide-react";

const LENS_COLORS: Record<string, { bg: string; color: string }> = {
  Literalist:            { bg: "rgba(148,163,184,0.12)", color: "#94a3b8" },
  Contradiction:         { bg: "rgba(249,115,22,0.12)",  color: "#fb923c" },
  Subtraction:           { bg: "rgba(244,63,94,0.10)",   color: "#fb7185" },
  "Wrong Medium":        { bg: "rgba(217,70,239,0.12)",  color: "#e879f9" },
  "Audience Inversion":  { bg: "rgba(20,184,166,0.12)",  color: "#2dd4bf" },
  "Material Honesty":    { bg: "rgba(212,168,83,0.12)",  color: "#d4a853" },
  "Narrative Compression": { bg: "rgba(59,130,246,0.12)", color: "#60a5fa" },
  "Emotional Extremity": { bg: "rgba(239,68,68,0.12)",   color: "#f87171" },
  "Cultural Counter-Signal": { bg: "rgba(16,185,129,0.12)", color: "#34d399" },
  "Constraint Injection":{ bg: "rgba(139,92,246,0.12)",  color: "#a78bfa" },
  "Found Object":        { bg: "rgba(245,158,11,0.12)",  color: "#fbbf24" },
  "Restraint Maximalism":{ bg: "rgba(99,102,241,0.12)",  color: "#818cf8" },
  "Hybrid Concept":      { bg: "rgba(99,102,241,0.08)",  color: "#a5b4fc" },
};

export default function SavedPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);
  const [concepts, setConcepts] = useState<ConceptRead[]>([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  useEffect(() => {
    api.getConcepts(projectId)
      .then(cs => setConcepts(cs.filter(c => c.status === "saved")))
      .finally(() => setLoading(false));
  }, [projectId]);

  async function downloadExport(type: "json" | "markdown") {
    setExporting(true);
    try {
      const blob = type === "json"
        ? await api.exportJson(projectId)
        : await api.exportMarkdown(projectId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `project_${projectId}_board.${type === "json" ? "json" : "md"}`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Loading saved concepts…</p>
    </div>
  );

  return (
    <div className="page-shell">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <Link href={`/projects/${projectId}/concepts`} className="btn-ghost" style={{ marginBottom: "12px", display: "inline-flex" }}>
            <ArrowLeft size={14} /> Back to Board
          </Link>
          <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Saved Concepts
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
            {concepts.length} concept{concepts.length !== 1 ? "s" : ""} saved for development
          </p>
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
          <Link href={`/projects/${projectId}/present`} className="btn-primary" style={{ fontSize: "13px" }}>
            <Presentation size={14} /> Present
          </Link>
          <button className="btn-secondary" style={{ fontSize: "13px" }} onClick={() => downloadExport("markdown")} disabled={exporting}>
            <FileText size={14} /> Export MD
          </button>
          <button className="btn-secondary" style={{ fontSize: "13px" }} onClick={() => downloadExport("json")} disabled={exporting}>
            <Download size={14} /> Export JSON
          </button>
        </div>
      </div>

      {concepts.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">
            <BookmarkCheck size={24} style={{ color: "var(--text-muted)" }} />
          </div>
          <h3 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "8px" }}>No saved concepts yet</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "24px" }}>
            Go to the concept board and save the directions you want to develop further.
          </p>
          <Link href={`/projects/${projectId}/concepts`} className="btn-primary">
            ← Concept Board
          </Link>
        </div>
      ) : (
        <div className="concept-grid">
          {concepts.map(c => {
            const ls = LENS_COLORS[c.style_category] ?? LENS_COLORS["Hybrid Concept"];
            return (
              <Link key={c.id} href={`/projects/${projectId}/concepts/${c.id}`} style={{ textDecoration: "none" }}>
                <div className="glass-card" style={{ padding: "0", overflow: "hidden", cursor: "pointer" }}>
                  <div style={{ height: "4px", background: `linear-gradient(90deg, ${ls.color}90, ${ls.color}20)` }} />
                  <div style={{ padding: "20px" }}>
                    <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "12px" }}>
                      <span className="lens-tag" style={{ background: ls.bg, color: ls.color }}>
                        {c.style_category}
                      </span>
                      <span style={{ padding: "3px 9px", borderRadius: "6px", background: "rgba(52,211,153,0.1)", color: "var(--success)", border: "1px solid rgba(52,211,153,0.2)", fontSize: "11px", fontWeight: 600 }}>
                        ✓ Saved
                      </span>
                    </div>
                    <h3 style={{ fontSize: "16px", fontWeight: 700, marginBottom: "8px", lineHeight: 1.35 }}>{c.title}</h3>
                    <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65, marginBottom: "14px" }}>
                      {c.main_visual_idea}
                    </p>
                    {c.scores && (
                      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                        <span className="score-chip novelty">N {Math.round(c.scores.novelty_score * 100)}%</span>
                        <span className="score-chip feasibility">F {Math.round(c.scores.feasibility_score * 100)}%</span>
                        <span className="score-chip fidelity">L {Math.round(c.scores.lens_fidelity_score * 100)}%</span>
                      </div>
                    )}
                    <p style={{ marginTop: "14px", fontSize: "12px", color: "var(--accent)", opacity: 0.7 }}>
                      View full reasoning →
                    </p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
