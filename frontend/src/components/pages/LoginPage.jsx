import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { destinationFor, getWorkspaceMode } from "../../context/workspaceMode";
import "./LoginPage.css";
import "./AuthFlowPages.css";

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
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  // Set by verify-email and reset-password so the outcome of those flows is
  // confirmed on the screen the user lands on, not lost in the redirect.
  const notice = location.state?.notice || "";

  const from = location.state?.from;

  async function handleSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      await signIn({ email, password, rememberMe });
      // First sign-in on this device: ask which experience they want before
      // dropping them somewhere. Afterwards the stored preference decides, so
      // the question is asked once and not on every login.
      const mode = getWorkspaceMode();
      if (!mode) {
        navigate("/choose-role", { replace: true, state: { from } });
      } else {
        navigate(destinationFor(mode, from), { replace: true });
      }
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
          {notice && !error && (
            <p className="nx-auth-ok" role="status">
              {notice}
            </p>
          )}
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

          <label className="nx-remember">
            <input
              type="checkbox"
              checked={rememberMe}
              onChange={(event) => setRememberMe(event.target.checked)}
            />
            <span>Keep me signed in on this device</span>
          </label>

          <div className="nx-login-options">
            <Link to="/forgot-password">Forgot password?</Link>
            <Link to="/register">Create an account</Link>
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
