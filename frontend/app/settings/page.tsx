"use client";
import { useEffect, useState } from "react";
import { api, ProviderSettings, TestResult } from "@/lib/api";
import { CheckCircle, XCircle, Loader, Cpu, Image, Database } from "lucide-react";

type TestStatus = "idle" | "testing" | "ok" | "fail";

export default function SettingsPage() {
  const [settings, setSettings] = useState<ProviderSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [llmTest, setLlmTest] = useState<TestStatus>("idle");
  const [imgTest, setImgTest] = useState<TestStatus>("idle");
  const [llmResult, setLlmResult] = useState<TestResult | null>(null);
  const [imgResult, setImgResult] = useState<TestResult | null>(null);

  useEffect(() => {
    api.getProviderSettings()
      .then(setSettings)
      .finally(() => setLoading(false));
  }, []);

  async function testLLM() {
    if (!settings) return;
    setLlmTest("testing");
    try {
      const r = await api.testLLM(settings.llm_provider);
      setLlmResult(r);
      setLlmTest(r.success ? "ok" : "fail");
    } catch {
      setLlmTest("fail");
    }
  }

  async function testImage() {
    if (!settings) return;
    setImgTest("testing");
    try {
      const r = await api.testImage(settings.image_provider);
      setImgResult(r);
      setImgTest(r.success ? "ok" : "fail");
    } catch {
      setImgTest("fail");
    }
  }

  if (loading) return (
    <div className="page-shell" style={{ textAlign: "center", paddingTop: "80px" }}>
      <div className="spinner" style={{ margin: "0 auto 16px" }} />
      <p style={{ color: "var(--text-secondary)" }}>Loading settings…</p>
    </div>
  );

  if (!settings) return (
    <div className="page-shell">
      <p style={{ color: "var(--danger)" }}>Could not load settings. Is the backend running?</p>
    </div>
  );

  return (
    <div className="page-shell" style={{ maxWidth: "800px" }}>
      <h1 style={{ fontSize: "28px", fontFamily: "'Playfair Display', serif", fontWeight: 700, marginBottom: "6px" }}>Settings</h1>
      <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginBottom: "36px" }}>
        AI provider configuration and connection health.
      </p>

      {/* Provider Overview */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "14px", marginBottom: "32px" }}>
        <ProviderCard
          icon={<Cpu size={18} style={{ color: "#818cf8" }} />}
          label="LLM Provider"
          value={settings.llm_provider}
          model={settings.llm_model}
          accent="#818cf8"
        />
        <ProviderCard
          icon={<Image size={18} style={{ color: "#34d399" }} />}
          label="Image Provider"
          value={settings.image_provider}
          accent="#34d399"
        />
        <ProviderCard
          icon={<Database size={18} style={{ color: "#d4a853" }} />}
          label="Embedding Provider"
          value={settings.embedding_provider}
          accent="#d4a853"
        />
      </div>

      {/* Connection Tests */}
      <div className="glass-card" style={{ padding: "28px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: "20px" }}>
          Connection Tests
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <TestRow
            label={`LLM — ${settings.llm_provider} (${settings.llm_model})`}
            status={llmTest}
            result={llmResult}
            onTest={testLLM}
          />
          <TestRow
            label={`Image — ${settings.image_provider}`}
            status={imgTest}
            result={imgResult}
            onTest={testImage}
          />
        </div>
      </div>

      {/* Available providers */}
      <div className="glass-card" style={{ padding: "28px", marginBottom: "24px" }}>
        <h2 style={{ fontSize: "14px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "var(--text-muted)", marginBottom: "20px" }}>
          Available Providers
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "20px" }}>
          <ProviderList label="LLM" items={settings.available_llm_providers} active={settings.llm_provider} />
          <ProviderList label="Image" items={settings.available_image_providers} active={settings.image_provider} />
          <ProviderList label="Embedding" items={settings.available_embedding_providers} active={settings.embedding_provider} />
        </div>
      </div>

      {/* Notice */}
      <div style={{ padding: "16px", borderRadius: "10px", background: "rgba(99,102,241,0.06)", border: "1px solid rgba(99,102,241,0.15)", fontSize: "13px", color: "var(--text-secondary)", lineHeight: 1.65 }}>
        <strong style={{ color: "#a5b4fc" }}>Configuration:</strong> Provider selection is controlled via environment variables in your backend <code style={{ background: "rgba(255,255,255,0.05)", padding: "1px 5px", borderRadius: "4px" }}>.env</code> file.
        Set <code style={{ background: "rgba(255,255,255,0.05)", padding: "1px 5px", borderRadius: "4px" }}>LLM_PROVIDER</code>, <code style={{ background: "rgba(255,255,255,0.05)", padding: "1px 5px", borderRadius: "4px" }}>IMAGE_PROVIDER</code>, and <code style={{ background: "rgba(255,255,255,0.05)", padding: "1px 5px", borderRadius: "4px" }}>GEMINI_API_KEY</code>.
      </div>
    </div>
  );
}

function ProviderCard({ icon, label, value, model, accent }: { icon: React.ReactNode; label: string; value: string; model?: string; accent: string }) {
  return (
    <div className="glass-card" style={{ padding: "18px", borderColor: `${accent}22` }}>
      <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "12px" }}>
        {icon}
        <span style={{ fontSize: "12px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)" }}>{label}</span>
      </div>
      <div style={{ fontSize: "15px", fontWeight: 600, color: accent, marginBottom: model ? "4px" : 0 }}>{value}</div>
      {model && <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>{model}</div>}
    </div>
  );
}

function TestRow({ label, status, result, onTest }: { label: string; status: TestStatus; result: TestResult | null; onTest: () => void }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "12px", padding: "12px 14px", borderRadius: "10px", background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", flexWrap: "wrap" }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: "13px", fontWeight: 500, marginBottom: "2px" }}>{label}</div>
        {result && (
          <div style={{ fontSize: "12px", color: result.success ? "var(--success)" : "var(--danger)" }}>
            {result.message} {result.latency_ms > 0 && `· ${result.latency_ms}ms`}
          </div>
        )}
      </div>
      {status === "ok" && <CheckCircle size={16} style={{ color: "var(--success)", flexShrink: 0 }} />}
      {status === "fail" && <XCircle size={16} style={{ color: "var(--danger)", flexShrink: 0 }} />}
      {status === "testing" && <Loader size={16} style={{ color: "var(--accent)", animation: "spin 1s linear infinite", flexShrink: 0 }} />}
      <button
        className="btn-secondary"
        style={{ fontSize: "12px", padding: "6px 14px" }}
        onClick={onTest}
        disabled={status === "testing"}
      >
        {status === "testing" ? "Testing…" : "Test"}
      </button>
    </div>
  );
}

function ProviderList({ label, items, active }: { label: string; items: string[]; active: string }) {
  return (
    <div>
      <div style={{ fontSize: "11px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-muted)", marginBottom: "10px" }}>{label}</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
        {items.map(item => (
          <div key={item} style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "13px", color: item === active ? "var(--text-primary)" : "var(--text-muted)" }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: item === active ? "var(--success)" : "var(--border)", flexShrink: 0 }} />
            {item}
            {item === active && <span style={{ fontSize: "10px", color: "var(--success)" }}>active</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
