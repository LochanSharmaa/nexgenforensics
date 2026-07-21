"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead } from "@/lib/api";
import { AlertTriangle, ArrowLeft, Lightbulb, RefreshCw } from "lucide-react";

export default function MissingPage({ params }: { params: Promise<{ id: string }> }) {
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
      <p style={{ color: "var(--text-secondary)" }}>Analysing missing opportunities…</p>
    </div>
  );

  // Derive lenses used and absent
  const usedLenses = [...new Set(concepts.map(c => c.style_category))];
  const allLenses = [
    "Literalist", "Contradiction", "Subtraction", "Wrong Medium", "Audience Inversion",
    "Material Honesty", "Narrative Compression", "Emotional Extremity",
    "Cultural Counter-Signal", "Constraint Injection", "Found Object", "Restraint Maximalism",
    "Time Inversion", "Scale Disruption", "Process Exposure", "Borrowed Vernacular",
    "Negative Space", "Texture Subversion", "Iconographic Reversal", "Anthropomorphism",
  ];
  const unusedLenses = allLenses.filter(l => !usedLenses.includes(l));

  // Collect all risk flags from reasoning chains
  const riskFlags = concepts
    .filter(c => c.reasoning_chain?.risk_flag)
    .map(c => ({ title: c.title, risk: c.reasoning_chain!.risk_flag, lens: c.style_category }));

  // Low-scoring concepts
  const lowFeasibility = concepts.filter(c => c.scores && c.scores.feasibility_score < 0.5);
  const lowFidelity = concepts.filter(c => c.scores && c.scores.lens_fidelity_score < 0.5);

  return (
    <div className="page-shell" style={{ maxWidth: "900px" }}>
      {/* Header */}
      <Link href={`/projects/${projectId}/concepts`} className="btn-ghost" style={{ marginBottom: "28px", display: "inline-flex" }}>
        <ArrowLeft size={14} /> Back to Board
      </Link>
      <div style={{ marginBottom: "32px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "4px 12px", borderRadius: "20px", background: "rgba(249,115,22,0.1)", border: "1px solid rgba(249,115,22,0.25)", fontSize: "12px", color: "var(--dissent)", fontWeight: 600, marginBottom: "14px" }}>
          <AlertTriangle size={11} /> Contradiction Engine Output
        </div>
        <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "8px" }}>
          Missing Opportunities
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", lineHeight: 1.7 }}>
          Directions the brief allowed but the generation didn't explore. Use these to seed additional concepts or challenge the current directions.
        </p>
      </div>

      {/* Unused lenses */}
      <div className="glass-card" style={{ padding: "28px", marginBottom: "20px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "20px" }}>
          <Lightbulb size={16} style={{ color: "#fbbf24" }} />
          <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            Unused Reasoning Lenses
          </h2>
        </div>
        {unusedLenses.length === 0 ? (
          <p style={{ fontSize: "13px", color: "var(--success)" }}>✓ All known lenses were explored in this generation run.</p>
        ) : (
          <>
            <p style={{ fontSize: "13px", color: "var(--text-secondary)", marginBottom: "16px" }}>
              These lenses were not selected by the CD Agent. They may offer productive angles worth manually requesting.
            </p>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              {unusedLenses.map(l => (
                <span key={l} style={{ padding: "5px 12px", borderRadius: "20px", background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)", fontSize: "12px", color: "var(--text-secondary)" }}>
                  {l}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Risk flags from reasoning chains */}
      {riskFlags.length > 0 && (
        <div className="glass-card" style={{ padding: "28px", marginBottom: "20px", background: "rgba(244,63,94,0.03)", borderColor: "rgba(244,63,94,0.12)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "20px" }}>
            <AlertTriangle size={16} style={{ color: "var(--danger)" }} />
            <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--danger)" }}>
              Risk Flags Identified
            </h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
            {riskFlags.map((r, i) => (
              <div key={i} style={{ padding: "14px 16px", borderRadius: "10px", background: "rgba(244,63,94,0.05)", border: "1px solid rgba(244,63,94,0.15)" }}>
                <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px" }}>
                  <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--danger)", textTransform: "uppercase" }}>{r.lens}</span>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>{r.title}</span>
                </div>
                <p style={{ fontSize: "13px", color: "#fca5a5", lineHeight: 1.65 }}>{r.risk}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Low feasibility */}
      {lowFeasibility.length > 0 && (
        <div className="glass-card" style={{ padding: "28px", marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <RefreshCw size={16} style={{ color: "#fbbf24" }} />
            <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
              Low Feasibility — Consider Refining
            </h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {lowFeasibility.map(c => (
              <Link key={c.id} href={`/projects/${projectId}/concepts/${c.id}`} style={{ textDecoration: "none" }}>
                <div style={{ padding: "12px 14px", borderRadius: "10px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>{c.title}</span>
                    <span style={{ marginLeft: "10px", fontSize: "11px", color: "var(--text-muted)" }}>{c.style_category}</span>
                  </div>
                  <span className="score-chip feasibility">F {Math.round((c.scores?.feasibility_score ?? 0) * 100)}%</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Low fidelity */}
      {lowFidelity.length > 0 && (
        <div className="glass-card" style={{ padding: "28px", marginBottom: "20px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
            <AlertTriangle size={16} style={{ color: "#d4a853" }} />
            <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
              Low Lens Fidelity — Lens Not Fully Honoured
            </h2>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {lowFidelity.map(c => (
              <Link key={c.id} href={`/projects/${projectId}/concepts/${c.id}`} style={{ textDecoration: "none" }}>
                <div style={{ padding: "12px 14px", borderRadius: "10px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>{c.title}</span>
                  <span className="score-chip fidelity">L {Math.round((c.scores?.lens_fidelity_score ?? 0) * 100)}%</span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      )}

      {riskFlags.length === 0 && lowFeasibility.length === 0 && lowFidelity.length === 0 && unusedLenses.length === 0 && (
        <div className="glass-card" style={{ padding: "32px", textAlign: "center" }}>
          <p style={{ color: "var(--success)", fontSize: "15px", marginBottom: "6px" }}>✓ Strong generation run</p>
          <p style={{ color: "var(--text-secondary)", fontSize: "13px" }}>No significant gaps or risks identified across this concept board.</p>
        </div>
      )}
    </div>
  );
}
