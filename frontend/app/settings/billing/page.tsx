"use client";
import Link from "next/link";
import { CreditCard, Zap, Shield, ArrowRight, Check } from "lucide-react";

const PLANS = [
  {
    id: "free",
    label: "Free",
    price: "$0",
    period: "forever",
    color: "#818cf8",
    features: [
      "3 concept generations / month",
      "10 concepts per run",
      "Mock image provider",
      "JSON + Markdown export",
      "All 20 reasoning lenses",
    ],
    cta: "Current Plan",
    disabled: true,
  },
  {
    id: "pro",
    label: "Pro",
    price: "$29",
    period: "/ month",
    color: "#34d399",
    popular: true,
    features: [
      "Unlimited generations",
      "10 concepts per run",
      "Imagen 3.0 image generation",
      "Priority API access",
      "All 20 reasoning lenses",
      "Genealogy tree",
      "Full reasoning export",
    ],
    cta: "Upgrade to Pro",
    disabled: false,
  },
  {
    id: "studio",
    label: "Studio",
    price: "$99",
    period: "/ month",
    color: "#d4a853",
    features: [
      "Everything in Pro",
      "Team workspace (5 seats)",
      "Custom lens configuration",
      "Webhook integrations",
      "Priority support",
      "API access",
      "SLA guarantee",
    ],
    cta: "Contact Sales",
    disabled: false,
  },
];

export default function BillingPage() {
  return (
    <div className="page-shell" style={{ maxWidth: "1000px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "52px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: "6px", padding: "4px 14px", borderRadius: "20px", background: "rgba(99,102,241,0.1)", border: "1px solid rgba(99,102,241,0.22)", fontSize: "12px", color: "#a5b4fc", fontWeight: 600, marginBottom: "18px" }}>
          <CreditCard size={11} /> Billing & Plans
        </div>
        <h1 style={{ fontSize: "36px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "12px" }}>
          Choose your plan
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: "15px", maxWidth: "480px", margin: "0 auto", lineHeight: 1.65 }}>
          The designer is always the final author. AI supplies the argument — pick the plan that matches your output.
        </p>
      </div>

      {/* Plans */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "20px", marginBottom: "48px" }}>
        {PLANS.map(plan => (
          <div
            key={plan.id}
            className="glass-card"
            style={{
              padding: "28px",
              border: plan.popular ? "1px solid rgba(52,211,153,0.35)" : undefined,
              background: plan.popular ? "rgba(52,211,153,0.04)" : undefined,
              position: "relative",
            }}
          >
            {plan.popular && (
              <div style={{ position: "absolute", top: "-12px", left: "50%", transform: "translateX(-50%)", background: "linear-gradient(135deg, #34d399, #059669)", color: "#fff", fontSize: "11px", fontWeight: 700, padding: "3px 12px", borderRadius: "12px", letterSpacing: "0.05em", whiteSpace: "nowrap" }}>
                MOST POPULAR
              </div>
            )}

            {/* Plan label */}
            <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "16px" }}>
              <Zap size={16} style={{ color: plan.color }} />
              <span style={{ fontSize: "14px", fontWeight: 700, color: plan.color, textTransform: "uppercase", letterSpacing: "0.06em" }}>{plan.label}</span>
            </div>

            {/* Price */}
            <div style={{ marginBottom: "20px" }}>
              <span style={{ fontSize: "36px", fontWeight: 700, color: "var(--text-primary)", fontFamily: "monospace" }}>{plan.price}</span>
              <span style={{ fontSize: "14px", color: "var(--text-muted)", marginLeft: "4px" }}>{plan.period}</span>
            </div>

            {/* Features */}
            <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginBottom: "24px" }}>
              {plan.features.map(f => (
                <div key={f} style={{ display: "flex", gap: "8px", alignItems: "flex-start" }}>
                  <Check size={13} style={{ color: plan.color, flexShrink: 0, marginTop: "2px" }} />
                  <span style={{ fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.5 }}>{f}</span>
                </div>
              ))}
            </div>

            {/* CTA */}
            <button
              className={plan.disabled ? "btn-secondary" : "btn-primary"}
              style={{
                width: "100%",
                justifyContent: "center",
                background: plan.disabled ? undefined : `linear-gradient(135deg, ${plan.color}cc, ${plan.color}88)`,
                boxShadow: plan.disabled ? "none" : `0 4px 20px ${plan.color}33`,
              }}
              disabled={plan.disabled}
            >
              {plan.cta} {!plan.disabled && <ArrowRight size={14} />}
            </button>
          </div>
        ))}
      </div>

      {/* Enterprise */}
      <div className="glass-card" style={{ padding: "32px", textAlign: "center", background: "linear-gradient(135deg, rgba(99,102,241,0.06), rgba(139,92,246,0.04))", borderColor: "rgba(99,102,241,0.15)" }}>
        <Shield size={24} style={{ color: "#818cf8", marginBottom: "12px" }} />
        <h2 style={{ fontSize: "20px", fontFamily: "'Playfair Display', serif", marginBottom: "8px" }}>Enterprise</h2>
        <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "20px", maxWidth: "440px", margin: "0 auto 20px" }}>
          Custom seat counts, private deployment, SSO, audit logging, custom lens training on your brand archive.
        </p>
        <a href="mailto:hello@sifs.ai" className="btn-secondary">
          Contact Sales <ArrowRight size={14} />
        </a>
      </div>

      {/* Back */}
      <div style={{ marginTop: "32px", textAlign: "center" }}>
        <Link href="/settings" className="btn-ghost">← Back to Settings</Link>
      </div>
    </div>
  );
}
