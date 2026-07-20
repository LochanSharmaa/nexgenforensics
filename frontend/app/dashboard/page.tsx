"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { api, Project } from "@/lib/api";
import { Plus, Clock, ArrowRight, FolderOpen, Sparkles } from "lucide-react";

const STATUS_COLORS: Record<string, string> = {
  pending: "#8892a4",
  brief_extracted: "#818cf8",
  concepts_generated: "#34d399",
};
const STATUS_LABELS: Record<string, string> = {
  pending: "New",
  brief_extracted: "Brief Ready",
  concepts_generated: "Board Complete",
};

export default function DashboardPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.getProjects()
      .then(setProjects)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="page-shell">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "40px" }}>
        <div>
          <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>
            Your Projects
          </h1>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px" }}>
            {projects.length} project{projects.length !== 1 ? "s" : ""} across your workspace
          </p>
        </div>
        <Link href="/projects/new" className="btn-primary">
          <Plus size={16} /> New Project
        </Link>
      </div>

      {/* State: Loading */}
      {loading && (
        <div style={{ textAlign: "center", padding: "80px", color: "var(--text-secondary)" }}>
          <div className="spinner" style={{ margin: "0 auto 16px" }} />
          <p>Loading projects…</p>
        </div>
      )}

      {/* State: Error */}
      {error && (
        <div className="glass-card" style={{ padding: "24px", borderColor: "rgba(244,63,94,0.3)", textAlign: "center" }}>
          <p style={{ color: "var(--danger)" }}>Failed to load projects: {error}</p>
          <p style={{ color: "var(--text-muted)", fontSize: "13px", marginTop: "8px" }}>
            Make sure the backend is running on http://localhost:8000
          </p>
        </div>
      )}

      {/* State: Empty */}
      {!loading && !error && projects.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">
            <FolderOpen size={24} style={{ color: "var(--text-muted)" }} />
          </div>
          <h3 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "8px" }}>No projects yet</h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "24px" }}>
            Enter your first rough imagination and let the engine expand it.
          </p>
          <Link href="/projects/new" className="btn-primary">
            <Sparkles size={15} /> Create First Project
          </Link>
        </div>
      )}

      {/* Projects grid */}
      {!loading && !error && projects.length > 0 && (
        <div className="concept-grid">
          {projects.map((p) => (
            <Link
              key={p.id}
              href={p.status === "concepts_generated" ? `/projects/${p.id}/concepts` : `/projects/${p.id}/brief`}
              style={{ textDecoration: "none" }}
            >
              <div className="glass-card" style={{ padding: "24px", cursor: "pointer" }}>
                {/* Status */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
                  <span style={{
                    fontSize: "11px", fontWeight: 600, padding: "3px 9px", borderRadius: "6px",
                    background: `${STATUS_COLORS[p.status]}18`,
                    color: STATUS_COLORS[p.status],
                    border: `1px solid ${STATUS_COLORS[p.status]}40`,
                  }}>
                    {STATUS_LABELS[p.status] ?? p.status}
                  </span>
                  <ArrowRight size={14} style={{ color: "var(--text-muted)" }} />
                </div>

                {/* Title */}
                <h3 style={{ fontSize: "16px", fontWeight: 600, marginBottom: "8px", lineHeight: 1.4 }}>{p.title}</h3>

                {/* Imagination preview */}
                <p style={{
                  fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.6,
                  display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden"
                }}>
                  {p.raw_imagination}
                </p>

                {/* Footer */}
                <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "16px", color: "var(--text-muted)", fontSize: "12px" }}>
                  <Clock size={11} />
                  {new Date(p.created_at).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}
                  {p.design_type && <span style={{ marginLeft: "6px", opacity: 0.7 }}>· {p.design_type}</span>}
                </div>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
