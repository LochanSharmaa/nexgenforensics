"use client";
import Link from "next/link";
import { Sparkles, ArrowRight, Brain, Layers, Zap, Eye } from "lucide-react";

const LENSES = [
  "Literalist", "Contradiction", "Subtraction", "Wrong Medium",
  "Audience Inversion", "Material Honesty", "Narrative Compression",
  "Emotional Extremity", "Cultural Counter-Signal", "Constraint Injection",
  "Found Object", "Restraint Maximalism"
];

const FEATURES = [
  {
    icon: <Brain size={22} style={{ color: "#818cf8" }} />,
    title: "Reasoning Chains, Not Just Images",
    desc: "Every concept exposes the exact thinking behind it: tension identified, design decision, consequence, and risk — not just a mood board.",
  },
  {
    icon: <Layers size={22} style={{ color: "#6ee7b7" }} />,
    title: "10 Dynamic Reasoning Lenses",
    desc: "Not fixed style buckets. A Lens Selector picks the 10 most productive and least predictable strategies for your specific brief.",
  },
  {
    icon: <Zap size={22} style={{ color: "#fcd34d" }} />,
    title: "Multi-Agent Debate",
    desc: "A Critic Panel challenges concepts. Dissents are shown alongside the concept — disagreement is a feature, not noise to be cleaned up.",
  },
  {
    icon: <Eye size={22} style={{ color: "#fb923c" }} />,
    title: "3-Axis Scoring",
    desc: "Novelty, Feasibility, and Lens Fidelity — three independent scores, never a fake composite. The designer decides what matters.",
  },
];

export default function LandingPage() {
  return (
    <div style={{ overflow: "hidden" }}>
      {/* Hero */}
      <section style={{ textAlign: "center", padding: "100px 24px 80px", maxWidth: "800px", margin: "0 auto" }}>
        <div style={{
          display: "inline-flex", alignItems: "center", gap: "8px",
          padding: "6px 16px", borderRadius: "20px",
          background: "rgba(99,102,241,0.12)", border: "1px solid rgba(99,102,241,0.25)",
          fontSize: "12px", color: "#a5b4fc", fontWeight: 600, marginBottom: "32px",
          letterSpacing: "0.05em", textTransform: "uppercase"
        }}>
          <Sparkles size={12} /> AI Creative Direction Assistant
        </div>

        <h1 style={{
          fontSize: "clamp(36px, 6vw, 64px)",
          fontFamily: "'Playfair Display', serif",
          fontWeight: 700,
          lineHeight: 1.15,
          letterSpacing: "-0.02em",
          marginBottom: "24px",
          background: "linear-gradient(135deg, #f0f2f7 0%, #a5b4fc 60%, #8b5cf6 100%)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}>
          One imagination.<br />Ten reasoning lenses.
        </h1>

        <p style={{ fontSize: "18px", color: "var(--text-secondary)", maxWidth: "560px", margin: "0 auto 40px", lineHeight: 1.7 }}>
          The designer's imagination is the input. A visible, disagreeing, evaluable chain of reasoning is the output. The image is just the receipt.
        </p>

        <div style={{ display: "flex", gap: "12px", justifyContent: "center", flexWrap: "wrap" }}>
          <Link href="/projects/new" className="btn-primary" style={{ padding: "14px 32px", fontSize: "15px" }}>
            Start Expanding <ArrowRight size={16} />
          </Link>
          <Link href="/dashboard" className="btn-secondary" style={{ padding: "14px 32px", fontSize: "15px" }}>
            View Dashboard
          </Link>
        </div>
      </section>

      {/* Lens ribbon */}
      <section style={{ padding: "40px 0", overflow: "hidden", borderTop: "1px solid var(--border)", borderBottom: "1px solid var(--border)" }}>
        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", justifyContent: "center", padding: "0 24px" }}>
          {LENSES.map((lens) => (
            <span key={lens} style={{
              padding: "5px 14px", borderRadius: "20px",
              background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)",
              fontSize: "12px", color: "var(--text-secondary)", fontWeight: 500,
            }}>
              {lens}
            </span>
          ))}
        </div>
        <p style={{ textAlign: "center", marginTop: "16px", fontSize: "12px", color: "var(--text-muted)" }}>
          12 creative reasoning lenses — selected dynamically per brief, never a fixed template
        </p>
      </section>

      {/* Features */}
      <section style={{ maxWidth: "1100px", margin: "0 auto", padding: "80px 24px" }}>
        <h2 style={{ textAlign: "center", fontSize: "28px", fontFamily: "'Playfair Display', serif", marginBottom: "48px", color: "var(--text-primary)" }}>
          Not a mood board generator
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "20px" }}>
          {FEATURES.map((f) => (
            <div key={f.title} className="glass-card" style={{ padding: "28px" }}>
              <div style={{ marginBottom: "14px" }}>{f.icon}</div>
              <h3 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "10px" }}>{f.title}</h3>
              <p style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section style={{ textAlign: "center", padding: "60px 24px 100px" }}>
        <div className="glass-card" style={{
          maxWidth: "600px", margin: "0 auto", padding: "48px",
          background: "linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.06) 100%)",
          border: "1px solid rgba(99,102,241,0.18)"
        }}>
          <h2 style={{ fontSize: "26px", fontFamily: "'Playfair Display', serif", marginBottom: "12px" }}>
            Ready to expand your imagination?
          </h2>
          <p style={{ color: "var(--text-secondary)", marginBottom: "28px", fontSize: "14px" }}>
            The designer is always the final author. AI supplies the argument, you make the art.
          </p>
          <Link href="/projects/new" className="btn-primary" style={{ padding: "13px 30px", fontSize: "15px" }}>
            Start Expanding <ArrowRight size={16} />
          </Link>
        </div>
      </section>
    </div>
  );
}
