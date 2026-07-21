"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead } from "@/lib/api";
import { ArrowLeft, GitBranch } from "lucide-react";

const LENS_COLORS: Record<string, string> = {
  Literalist: "#94a3b8", Contradiction: "#fb923c", Subtraction: "#fb7185",
  "Wrong Medium": "#e879f9", "Audience Inversion": "#2dd4bf",
  "Material Honesty": "#d4a853", "Narrative Compression": "#60a5fa",
  "Emotional Extremity": "#f87171", "Cultural Counter-Signal": "#34d399",
  "Constraint Injection": "#a78bfa", "Found Object": "#fbbf24",
  "Restraint Maximalism": "#818cf8", "Hybrid Concept": "#a5b4fc",
};

const STATUS_DOT: Record<string, string> = {
  active: "#818cf8", saved: "#34d399", rejected: "#4d5666", combined: "#fbbf24",
};

export default function GenealogyPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);
  const [concepts, setConcepts] = useState<ConceptRead[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getConcepts(projectId)
      .then(setConcepts)
      .finally(() => setLoading(false));
  }, [projectId]);

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Building genealogy…</p>
    </div>
  );

  // Group by parent / generation
  const roots = concepts.filter(c => !c.parent_id);
  const children = concepts.filter(c => !!c.parent_id);
  const byParent: Record<number, ConceptRead[]> = {};
  children.forEach(c => {
    if (!byParent[c.parent_id!]) byParent[c.parent_id!] = [];
    byParent[c.parent_id!].push(c);
  });

  const total = concepts.length;
  const saved = concepts.filter(c => c.status === "saved").length;
  const rejected = concepts.filter(c => c.status === "rejected").length;
  const combined = concepts.filter(c => c.style_category === "Hybrid Concept").length;

  return (
    <div className="page-shell">
      {/* Header */}
      <Link href={`/projects/${projectId}/concepts`} className="btn-ghost" style={{ marginBottom: "28px", display: "inline-flex" }}>
        <ArrowLeft size={14} /> Back to Board
      </Link>
      <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", marginBottom: "32px" }}>
        <GitBranch size={28} style={{ color: "#818cf8", marginTop: "4px", flexShrink: 0 }} />
        <div>
          <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Concept Genealogy
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
            {total} total concepts — {saved} saved, {rejected} rejected, {combined} combined
          </p>
        </div>
      </div>

      {/* Stats bar */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "12px", marginBottom: "32px" }}>
        {[
          { label: "Total", value: total, color: "#818cf8" },
          { label: "Saved", value: saved, color: "#34d399" },
          { label: "Rejected", value: rejected, color: "#4d5666" },
          { label: "Combined", value: combined, color: "#fbbf24" },
        ].map(s => (
          <div key={s.label} className="glass-card" style={{ padding: "16px", textAlign: "center" }}>
            <div style={{ fontSize: "28px", fontWeight: 700, color: s.color, fontFamily: "monospace" }}>{s.value}</div>
            <div style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "4px", textTransform: "uppercase", letterSpacing: "0.05em" }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Genealogy tree */}
      <div className="glass-card" style={{ padding: "28px" }}>
        <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: "24px" }}>
          Generation Tree
        </h2>

        {roots.length === 0 ? (
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>No concepts generated yet.</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {roots.map(root => {
              const kids = byParent[root.id] ?? [];
              const lensColor = LENS_COLORS[root.style_category] ?? "#818cf8";
              const dotColor = STATUS_DOT[root.status] ?? "#818cf8";
              return (
                <div key={root.id}>
                  {/* Root node */}
                  <Link href={`/projects/${projectId}/concepts/${root.id}`} style={{ textDecoration: "none" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 16px", borderRadius: "10px", background: "rgba(255,255,255,0.03)", border: `1px solid ${lensColor}30`, cursor: "pointer", transition: "background 0.2s" }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)"}
                    >
                      <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: dotColor, flexShrink: 0 }} />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap" }}>
                          <span style={{ fontSize: "14px", fontWeight: 600, color: "var(--text-primary)" }}>#{root.concept_number} {root.title}</span>
                          <span style={{ fontSize: "11px", padding: "2px 8px", borderRadius: "10px", background: `${lensColor}18`, color: lensColor, fontWeight: 600 }}>{root.style_category}</span>
                          {root.status === "saved" && <span style={{ fontSize: "11px", color: "#34d399" }}>✓ Saved</span>}
                          {root.status === "rejected" && <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>✗ Rejected</span>}
                        </div>
                        <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "2px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {root.main_visual_idea}
                        </p>
                      </div>
                      {root.scores && (
                        <div style={{ display: "flex", gap: "6px", flexShrink: 0 }}>
                          <span className="score-chip novelty">N {Math.round(root.scores.novelty_score * 100)}%</span>
                          <span className="score-chip feasibility">F {Math.round(root.scores.feasibility_score * 100)}%</span>
                        </div>
                      )}
                    </div>
                  </Link>

                  {/* Children */}
                  {kids.length > 0 && (
                    <div style={{ marginLeft: "28px", marginTop: "8px", borderLeft: "2px solid var(--border)", paddingLeft: "20px", display: "flex", flexDirection: "column", gap: "8px" }}>
                      {kids.map(child => {
                        const cl = LENS_COLORS[child.style_category] ?? "#818cf8";
                        const cd = STATUS_DOT[child.status] ?? "#818cf8";
                        return (
                          <Link key={child.id} href={`/projects/${projectId}/concepts/${child.id}`} style={{ textDecoration: "none" }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 14px", borderRadius: "10px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", cursor: "pointer" }}
                              onMouseEnter={e => (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.05)"}
                              onMouseLeave={e => (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.02)"}
                            >
                              <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: cd, flexShrink: 0 }} />
                              <div style={{ flex: 1, minWidth: 0 }}>
                                <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                                  <span style={{ fontSize: "13px", fontWeight: 500 }}>#{child.concept_number} {child.title}</span>
                                  <span style={{ fontSize: "10px", padding: "2px 6px", borderRadius: "8px", background: `${cl}15`, color: cl }}>
                                    {child.style_category === "Hybrid Concept" ? "Combined" : "Regen"}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </Link>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
