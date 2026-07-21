"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { api, ConceptRead } from "@/lib/api";
import { ArrowLeft, AlertTriangle, Brain, Camera, Lightbulb, Package, Shield } from "lucide-react";

function scoreColor(v: number) {
  if (v >= 4.0) return "var(--success)";
  if (v >= 3.0) return "#fbbf24";
  return "var(--danger)";
}

export default function ConceptReasoningPage({ params }: { params: Promise<{ id: string; conceptId: string }> }) {
  const { id, conceptId } = use(params);
  const projectId = parseInt(id);
  const cId = parseInt(conceptId);

  const [concept, setConcept] = useState<ConceptRead | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getConcepts(projectId)
      .then((cs) => setConcept(cs.find((c) => c.id === cId) ?? null))
      .finally(() => setLoading(false));
  }, [projectId, cId]);

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
    </div>
  );

  if (!concept) return (
    <div className="page-shell">
      <p style={{ color: "var(--danger)" }}>Concept not found.</p>
      <Link href={`/projects/${projectId}/concepts`} className="btn-secondary" style={{ marginTop: "16px" }}>← Back to board</Link>
    </div>
  );

  const { reasoning_chain: rc, scores, critic_notes } = concept;
  const hasDissent = critic_notes?.some((n) => n.is_dissent);

  return (
    <div className="page-shell" style={{ maxWidth: "860px" }}>
      {/* Back */}
      <Link href={`/projects/${projectId}/concepts`} className="btn-ghost" style={{ marginBottom: "28px", display: "inline-flex" }}>
        <ArrowLeft size={14} /> Back to Board
      </Link>

      {/* Header */}
      <div style={{ marginBottom: "32px" }}>
        <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "12px", flexWrap: "wrap" }}>
          <span style={{
            padding: "4px 12px", borderRadius: "20px", fontSize: "11px", fontWeight: 700,
            background: "rgba(99,102,241,0.12)", color: "#818cf8",
            textTransform: "uppercase", letterSpacing: "0.06em"
          }}>
            {concept.style_category}
          </span>
          {hasDissent && <span className="dissent-badge" style={{ background: "rgba(249,115,22,0.12)", color: "#fb923c", border: "1px solid rgba(249,115,22,0.2)" }}><AlertTriangle size={10} />Critic Dissent</span>}
          <span className="reference-notice">🔒 Reference only — final artwork belongs to the designer</span>
        </div>
        <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "10px" }}>
          {concept.title}
        </h1>
        <p style={{ fontSize: "16px", color: "var(--text-secondary)", lineHeight: 1.7 }}>
          {concept.main_visual_idea}
        </p>
      </div>

      {/* 3-Axis Scores */}
      {scores && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
          <h2 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "20px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            3-Axis Evaluation (1 to 5 Rubric)
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: "16px" }}>
            <ScoreAxis label="Novelty" value={scores.novelty_score} sublabel="vs other concepts + design history" />
            <ScoreAxis label="Feasibility" value={scores.feasibility_score} sublabel={scores.feasibility_reason ?? ""} />
            <ScoreAxis label="Lens Fidelity" value={scores.lens_fidelity_score} sublabel="How strictly the concept embodies the lens move" />
          </div>
        </div>
      )}

      {/* Reasoning Chain */}
      {rc && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "20px" }}>
            <Brain size={16} style={{ color: "#818cf8" }} />
            <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
              Reasoning Chain
            </h2>
          </div>
          <ChainStep label="Lens Applied" value={rc.lens_applied} color="#818cf8" />
          <ChainStep label="Tension Identified" value={rc.tension_identified} color="#fb923c" />
          <ChainStep label="Design Decision" value={rc.design_decision} color="#60a5fa" />
          <ChainStep label="Consequence" value={rc.consequence} color="#34d399" />
          <ChainStep label="Risk Flag" value={rc.risk_flag} color="#f87171" last />
        </div>
      )}

      {/* Visual Details */}
      <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "20px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Visual Direction
        </h2>
        <DetailRow label="Composition" value={concept.composition} />
        <DetailRow label="Background" value={concept.background} />
        <DetailRow label="Creative Twist" value={concept.creative_twist} />
        <DetailRow label="Typography" value={concept.typography_direction} />
        {concept.color_palette?.length > 0 && (
          <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "12px", padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
            <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", paddingTop: "2px" }}>Color Palette</span>
            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              {concept.color_palette.map((c) => (
                <span key={c} style={{ padding: "3px 10px", borderRadius: "6px", background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)", fontSize: "12px", color: "var(--text-secondary)" }}>{c}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Specialist Sub-panels */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "16px", marginBottom: "24px" }}>
        {concept.camera_language && (
          <SubPanel icon={<Camera size={15} style={{ color: "#60a5fa" }} />} title="Camera Language" value={concept.camera_language} />
        )}
        {concept.lighting_reasoning && (
          <SubPanel icon={<Lightbulb size={15} style={{ color: "#fbbf24" }} />} title="Lighting Reasoning" value={concept.lighting_reasoning} />
        )}
        {concept.material_reasoning && (
          <SubPanel icon={<Package size={15} style={{ color: "#34d399" }} />} title="Material Reasoning" value={concept.material_reasoning} />
        )}
      </div>

      {/* Metaphor */}
      {concept.metaphor_primary && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
          <h2 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "16px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            Visual Metaphor
          </h2>
          <p style={{ fontSize: "14px", color: "var(--text-primary)", marginBottom: "12px" }}>
            <strong>Chosen Metaphor:</strong> {concept.metaphor_primary}
          </p>
          {concept.metaphor_rejected && concept.metaphor_rejected.length > 0 && (
            <div style={{ marginTop: "12px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>Rejected Alternatives</span>
              {concept.metaphor_rejected.map((item, idx) => (
                <div key={idx} style={{ fontSize: "13px", color: "var(--text-muted)", paddingLeft: "8px", borderLeft: "2px solid var(--border)" }}>
                  <strong style={{ color: "var(--text-secondary)" }}>{item.metaphor}:</strong> {item.reason}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Anti-pattern */}
      {concept.anti_pattern && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px", background: "rgba(244,63,94,0.04)", borderColor: "rgba(244,63,94,0.15)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
            <Shield size={15} style={{ color: "var(--danger)" }} />
            <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--danger)" }}>
              Anti-Pattern: What Not To Do
            </h2>
          </div>
          <p style={{ fontSize: "14px", color: "#fca5a5", lineHeight: 1.65 }}>{concept.anti_pattern}</p>
        </div>
      )}

      {/* Execution Notes */}
      <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
          Designer Execution Notes
        </h2>
        <p style={{ fontSize: "14px", color: "var(--text-secondary)", lineHeight: 1.7 }}>{concept.designer_execution_notes}</p>
      </div>

      {/* Critic Notes / Dissent */}
      {critic_notes && critic_notes.length > 0 && (
        <div className="glass-card" style={{ padding: "24px", marginBottom: "24px" }}>
          <h2 style={{ fontSize: "14px", fontWeight: 700, marginBottom: "16px", textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)" }}>
            Critic Panel Notes
          </h2>
          {critic_notes.map((note, i) => (
            <div key={i} style={{
              padding: "14px 16px", borderRadius: "10px", marginBottom: "10px",
              background: note.is_dissent ? "rgba(249,115,22,0.08)" : "rgba(255,255,255,0.03)",
              border: `1px solid ${note.is_dissent ? "rgba(249,115,22,0.25)" : "var(--border)"}`,
            }}>
              <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px" }}>
                {note.is_dissent && <span className="dissent-badge" style={{ background: "rgba(249,115,22,0.12)", color: "#fb923c", border: "1px solid rgba(249,115,22,0.2)" }}><AlertTriangle size={10} />Dissent</span>}
                <span style={{ fontSize: "12px", fontWeight: 600, color: note.is_dissent ? "#fb923c" : "var(--text-secondary)" }}>
                  {note.agent_role}
                </span>
              </div>
              <p style={{ fontSize: "13px", color: note.is_dissent ? "#fdba74" : "var(--text-secondary)", lineHeight: 1.65 }}>
                {note.note_text}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Reference prompt */}
      <div className="glass-card" style={{ padding: "20px", borderColor: "rgba(99,102,241,0.15)" }}>
        <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "10px" }}>
          <span className="reference-notice">🔒 Reference only — final artwork belongs to the designer</span>
        </div>
        <p style={{ fontSize: "12px", color: "var(--text-muted)", fontFamily: "monospace", lineHeight: 1.6 }}>
          {concept.reference_image_prompt}
        </p>
      </div>
    </div>
  );
}

function ScoreAxis({ label, value, sublabel }: { label: string; value: number; sublabel: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: "28px", fontWeight: 700, color: scoreColor(value), fontFamily: "monospace", marginBottom: "4px" }}>
        {value.toFixed(1)}/5
      </div>
      <div style={{ fontSize: "13px", fontWeight: 600, marginBottom: "6px" }}>{label}</div>
      <div style={{ fontSize: "11px", color: "var(--text-muted)", lineHeight: 1.5 }}>{sublabel}</div>
    </div>
  );
}

function ChainStep({ label, value, color, last = false }: { label: string; value: string; color: string; last?: boolean }) {
  return (
    <div style={{ display: "flex", gap: "16px", marginBottom: last ? 0 : "16px" }}>
      <div style={{ flexShrink: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: "4px" }}>
        <div style={{ width: "8px", height: "8px", borderRadius: "50%", background: color, marginTop: "6px" }} />
        {!last && <div style={{ width: "1px", flex: 1, background: "var(--border)" }} />}
      </div>
      <div style={{ flex: 1, paddingBottom: last ? 0 : "8px" }}>
        <span style={{ fontSize: "11px", fontWeight: 700, color: color, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</span>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65, marginTop: "4px" }}>{value}</p>
      </div>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: "12px", padding: "12px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em", paddingTop: "2px" }}>{label}</span>
      <span style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.6 }}>{value}</span>
    </div>
  );
}

function SubPanel({ icon, title, value }: { icon: React.ReactNode; title: string; value: string }) {
  return (
    <div className="glass-card" style={{ padding: "18px" }}>
      <div style={{ display: "flex", gap: "8px", alignItems: "center", marginBottom: "10px" }}>
        {icon}
        <span style={{ fontSize: "12px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>{title}</span>
      </div>
      <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65 }}>{value}</p>
    </div>
  );
}
