"use client";
import { use, useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { api, ConceptRead, ImageJobRead } from "@/lib/api";
import { Image as ImageIcon, Zap, ArrowLeft, RefreshCw, AlertTriangle, CheckCircle, Clock, Link as LinkIcon } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL?.replace("/api", "") || "http://localhost:8000";

const STATUS_CONFIG = {
  queued:    { color: "#818cf8", icon: <Clock size={13} />,        label: "Queued"     },
  running:   { color: "#fbbf24", icon: <RefreshCw size={13} className="spin-anim" />, label: "Generating" },
  completed: { color: "#34d399", icon: <CheckCircle size={13} />,  label: "Done"       },
  failed:    { color: "#f87171", icon: <AlertTriangle size={13} />, label: "Failed"     },
};

export default function ImagesPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const projectId = parseInt(id);

  const [concepts, setConcepts] = useState<ConceptRead[]>([]);
  const [jobs, setJobs] = useState<Record<number, ImageJobRead>>({});
  const [loading, setLoading] = useState(true);
  const [batchLoading, setBatchLoading] = useState(false);

  useEffect(() => {
    api.getConcepts(projectId)
      .then(cs => setConcepts(cs.filter(c => c.status !== "rejected")))
      .finally(() => setLoading(false));
  }, [projectId]);

  // Poll active jobs
  const pollJobs = useCallback(async () => {
    const activeJobs = Object.values(jobs).filter(j => j.status === "queued" || j.status === "running");
    for (const job of activeJobs) {
      try {
        const updated = await api.getImageJob(job.id);
        setJobs(prev => ({ ...prev, [job.concept_id]: updated }));
      } catch {}
    }
  }, [jobs]);

  useEffect(() => {
    const interval = setInterval(pollJobs, 3000);
    return () => clearInterval(interval);
  }, [pollJobs]);

  async function generateOne(conceptId: number) {
    const job = await api.generateImage(conceptId);
    setJobs(prev => ({ ...prev, [conceptId]: job }));
  }

  async function generateBatch() {
    setBatchLoading(true);
    try {
      const ids = concepts.map(c => c.id);
      const newJobs = await api.generateImageBatch(projectId, ids);
      const byConceptId: Record<number, ImageJobRead> = {};
      newJobs.forEach(j => { byConceptId[j.concept_id] = j; });
      setJobs(prev => ({ ...prev, ...byConceptId }));
    } finally {
      setBatchLoading(false);
    }
  }

  function handleUseAsReference(title: string, url: string) {
    const fullUrl = url.startsWith("http") ? url : `${API_BASE}${url}`;
    navigator.clipboard.writeText(fullUrl).then(() => {
      alert(`"Use as reference" action triggered.\nCopied reference image URL for "${title}" to your clipboard.\n\nNotice: Reference only — final artwork belongs to the designer.`);
    });
  }

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Loading image board…</p>
    </div>
  );

  return (
    <div className="page-shell">
      <style>{`.spin-anim { animation: spin 1s linear infinite; }`}</style>
      
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "32px", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <Link href={`/projects/${projectId}/concepts`} className="btn-ghost" style={{ marginBottom: "12px", display: "inline-flex" }}>
            <ArrowLeft size={14} /> Back to Board
          </Link>
          <h1 style={{ fontSize: "32px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Reference Images
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "8px" }}>
            AI-generated reference imagery — for ideation only. The designer creates the final work.
          </p>
          <span className="reference-notice" style={{ display: "inline-flex" }}>
            🔒 Reference only — final artwork belongs to the designer
          </span>
        </div>
        <button className="btn-primary" onClick={generateBatch} disabled={batchLoading}>
          {batchLoading ? <><span className="spinner" />Generating all…</> : <><Zap size={15} />Generate All</>}
        </button>
      </div>

      {/* Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "20px" }}>
        {concepts.map(c => {
          const job = jobs[c.id];
          const status = job?.status;
          const sc = STATUS_CONFIG[status as keyof typeof STATUS_CONFIG];

          return (
            <div key={c.id} className="glass-card" style={{ padding: "0", overflow: "hidden" }}>
              {/* Image area */}
              <div style={{
                height: "220px",
                background: "rgba(255,255,255,0.02)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                overflow: "hidden",
                borderBottom: "1px solid var(--border)"
              }}>
                {status === "completed" && job.image_url ? (
                  <>
                    <img
                      src={job.image_url.startsWith("http") ? job.image_url : `${API_BASE}${job.image_url}`}
                      alt={c.title}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                    {/* Watermark overlay */}
                    <div style={{
                      position: "absolute",
                      inset: 0,
                      background: "repeating-linear-gradient(45deg, rgba(0,0,0,0.06), rgba(0,0,0,0.06) 15px, transparent 15px, transparent 30px)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      pointerEvents: "none"
                    }}>
                      <span style={{
                        background: "rgba(10, 10, 12, 0.8)",
                        color: "rgba(255, 255, 255, 0.7)",
                        padding: "6px 12px",
                        fontSize: "10px",
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.08em",
                        border: "1px solid rgba(255,255,255,0.15)",
                        borderRadius: "4px"
                      }}>
                        Reference Only — belongs to designer
                      </span>
                    </div>
                  </>
                ) : status === "running" || status === "queued" ? (
                  <div style={{ textAlign: "center" }}>
                    <div className="spinner" style={{ margin: "0 auto 12px" }} />
                    <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                      {status === "queued" ? "Queued…" : "Generating…"}
                    </p>
                  </div>
                ) : status === "failed" ? (
                  <div style={{ textAlign: "center", padding: "16px" }}>
                    <AlertTriangle size={24} style={{ color: "var(--danger)", marginBottom: "8px" }} />
                    <p style={{ fontSize: "12px", color: "var(--danger)" }}>
                      {job?.error_message_sanitized ?? "Generation failed"}
                    </p>
                  </div>
                ) : (
                  <div style={{ textAlign: "center" }}>
                    <ImageIcon size={32} style={{ color: "var(--text-muted)", marginBottom: "12px" }} />
                    <p style={{ fontSize: "12px", color: "var(--text-muted)" }}>No image generated yet</p>
                  </div>
                )}
              </div>

              {/* Card body */}
              <div style={{ padding: "16px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "8px" }}>
                  <h3 style={{ fontSize: "15px", fontWeight: 600, lineHeight: 1.35 }}>{c.title}</h3>
                  {sc && (
                    <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", fontSize: "11px", color: sc.color, fontWeight: 600, flexShrink: 0, marginLeft: "8px" }}>
                      {sc.icon} {sc.label}
                    </span>
                  )}
                </div>

                <p style={{ fontSize: "12px", color: "var(--text-muted)", lineHeight: 1.55, marginBottom: "14px", fontStyle: "italic" }}>
                  {c.reference_image_prompt?.slice(0, 120)}{(c.reference_image_prompt?.length ?? 0) > 120 ? "…" : ""}
                </p>

                <div style={{ display: "flex", gap: "8px" }}>
                  {status === "completed" && job.image_url ? (
                    <button
                      className="btn-success"
                      style={{ flex: 1.3, justifyContent: "center", fontSize: "12px", padding: "7px 12px" }}
                      onClick={() => handleUseAsReference(c.title, job.image_url!)}
                    >
                      <LinkIcon size={12} /> Use as reference
                    </button>
                  ) : (
                    <button
                      className="btn-secondary"
                      style={{ flex: 1.3, justifyContent: "center", fontSize: "12px", padding: "7px 12px" }}
                      onClick={() => generateOne(c.id)}
                      disabled={status === "queued" || status === "running"}
                    >
                      {status === "running" ? <><span className="spinner" />Working</> : <><Zap size={12} />Generate</>}
                    </button>
                  )}
                  <Link
                    href={`/projects/${projectId}/concepts/${c.id}`}
                    className="btn-ghost"
                    style={{ fontSize: "12px", border: "1px solid var(--border)", borderRadius: "8px", flex: 0.7, textAlign: "center" }}
                  >
                    Reasoning
                  </Link>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
