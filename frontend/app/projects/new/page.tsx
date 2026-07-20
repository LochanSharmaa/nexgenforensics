"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { Sparkles, ArrowRight, Lightbulb } from "lucide-react";

const EXAMPLE = "A dark luxury perfume poster with a bottle floating above black water, moonlight reflection, blue-black premium mood, silver typography, minimal text.";

export default function NewProjectPage() {
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [imagination, setImagination] = useState("");
  const [designType, setDesignType] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const wordCount = imagination.trim().split(/\s+/).filter(Boolean).length;
  const isThin = wordCount > 0 && wordCount < 15;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!imagination.trim() || !title.trim()) return;

    setLoading(true);
    setError("");
    try {
      const project = await api.createProject({
        title: title.trim(),
        raw_imagination: imagination.trim(),
        design_type: designType.trim() || undefined,
      });
      router.push(`/projects/${project.id}/brief`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create project");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell" style={{ maxWidth: "720px" }}>
      {/* Header */}
      <div style={{ marginBottom: "40px" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "6px",
          fontSize: "12px", color: "#a5b4fc", fontWeight: 600,
          padding: "4px 12px", borderRadius: "20px",
          background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.2)",
          marginBottom: "16px"
        }}>
          <Sparkles size={11} /> New Project
        </div>
        <h1 style={{ fontSize: "32px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "10px" }}>
          Enter your imagination
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "15px", lineHeight: 1.65 }}>
          Write your rough idea, concept, or mood — as messy or as detailed as it is. The engine will expand it into 10 distinct reasoning directions.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {/* Project Title */}
        <div style={{ marginBottom: "20px" }}>
          <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
            Project Title *
          </label>
          <input
            className="input-field"
            type="text"
            placeholder="e.g. Noir Perfume Campaign"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>

        {/* Raw Imagination */}
        <div style={{ marginBottom: "8px" }}>
          <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
            Raw Imagination *
          </label>
          <textarea
            className="input-field"
            placeholder="Describe your idea as freely as you like…"
            value={imagination}
            onChange={(e) => setImagination(e.target.value)}
            rows={6}
            required
          />
        </div>

        {/* Word count + thin warning */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
          <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>{wordCount} words</span>
          {isThin && (
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#fbbf24" }}>
              <Lightbulb size={12} />
              Try adding more detail — colour, mood, subject, or purpose — for richer concepts.
            </div>
          )}
        </div>

        {/* Design Type (optional) */}
        <div style={{ marginBottom: "28px" }}>
          <label style={{ display: "block", fontSize: "13px", fontWeight: 600, marginBottom: "8px", color: "var(--text-secondary)" }}>
            Design Type <span style={{ opacity: 0.5, fontWeight: 400 }}>(optional)</span>
          </label>
          <input
            className="input-field"
            type="text"
            placeholder="e.g. poster, packaging, social ad, book cover…"
            value={designType}
            onChange={(e) => setDesignType(e.target.value)}
          />
        </div>

        {error && (
          <div style={{ marginBottom: "20px", padding: "12px 16px", borderRadius: "10px", background: "rgba(244,63,94,0.08)", border: "1px solid rgba(244,63,94,0.2)", color: "var(--danger)", fontSize: "13px" }}>
            {error}
          </div>
        )}

        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <button type="submit" className="btn-primary" disabled={loading || !title.trim() || !imagination.trim()}>
            {loading ? <><span className="spinner" />Extracting Brief…</> : <>Extract Creative Brief <ArrowRight size={15} /></>}
          </button>
        </div>
      </form>

      {/* Example */}
      <div className="glass-card" style={{ padding: "20px", marginTop: "40px" }}>
        <p style={{ fontSize: "12px", color: "var(--text-muted)", fontWeight: 600, marginBottom: "8px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Example imagination
        </p>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65, fontStyle: "italic" }}>
          "{EXAMPLE}"
        </p>
        <button
          className="btn-ghost"
          style={{ marginTop: "10px", fontSize: "12px" }}
          type="button"
          onClick={() => { setImagination(EXAMPLE); setTitle("Dark Luxury Perfume Campaign"); }}
        >
          Use this example →
        </button>
      </div>
    </div>
  );
}
