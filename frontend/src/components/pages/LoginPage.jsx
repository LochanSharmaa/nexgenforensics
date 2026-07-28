import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import "./LoginPage.css";

const accessItems = [
  "Case management and search history",
  "iMATCH biometric search and 1:1 comparison",
  "Hash-chained audit trail",
];

const trustItems = ["Encrypted templates", "Role-based access", "Every search audited"];

export function LoginPage() {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [tenant, setTenant] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const destination = location.state?.from?.pathname || "/workspace";

  async function handleSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await signIn({ email, password, tenant });
      navigate(destination, { replace: true });
    } catch (signInError) {
      setError(signInError.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="nx-login-page" id="top">
      <div className="nx-login-copy">
        <p className="nx-kicker">Secure Access</p>
        <h1>NexGen iMATCH Login</h1>
        <p>
          Sign in with your workspace credentials. Access is scoped to your organisation, and
          every biometric operation you perform is recorded against your account.
        </p>

        <div className="nx-login-access-list" aria-label="Available workspaces">
          {accessItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>

      <div className="nx-login-panel" aria-label="iMATCH login form">
        <div className="nx-login-panel-header">
          <span>NGF</span>
          <div>
            <strong>Authorized Login</strong>
            <small>Investigation workspace</small>
          </div>
        </div>

        <form className="nx-login-form" onSubmit={handleSubmit}>
          {error && (
            <p className="nx-login-error" role="alert">
              {error}
            </p>
          )}

          <label>
            <span>Email Address</span>
            <input
              type="email"
              name="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              placeholder="analyst@agency.gov"
              autoComplete="email"
            />
          </label>

          <label>
            <span>Password</span>
            <input
              type="password"
              name="password"
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </label>

          <label>
            <span>Organisation (only if prompted)</span>
            <input
              type="text"
              name="tenant"
              value={tenant}
              onChange={(event) => setTenant(event.target.value)}
              placeholder="tenant-slug"
              autoComplete="organization"
            />
          </label>

          <div className="nx-login-options">
            <a href="/contact">Need access?</a>
          </div>

          <button type="submit" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>

        <div className="nx-login-trust-row" aria-label="Security features">
          {trustItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

export default LoginPage;
