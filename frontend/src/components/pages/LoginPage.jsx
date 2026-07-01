import "./LoginPage.css";

const accessItems = [
  "Evidence review workspace",
  "iMatch biometric demo",
  "Case intelligence dashboard",
];

const trustItems = ["Encrypted session", "Role-based access", "Audit-ready activity"];

export function LoginPage() {
  return (
    <section className="nx-login-page" id="top">
      <div className="nx-login-copy">
        <p className="nx-kicker">Secure Access</p>
        <h1>NexGen Forensics Login</h1>
        <p>
          Enter your authorized workspace credentials to continue into NexGen forensic tools,
          evidence workflows, and demo environments.
        </p>

        <div className="nx-login-access-list" aria-label="Available NexGen workspaces">
          {accessItems.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </div>

      <div className="nx-login-panel" aria-label="NexGen Forensics login form">
        <div className="nx-login-panel-header">
          <span>NGF</span>
          <div>
            <strong>Authorized Login</strong>
            <small>Investigation workspace</small>
          </div>
        </div>

        <form className="nx-login-form">
          <label>
            <span>Email Address</span>
            <input type="email" name="email" placeholder="analyst@agency.gov" autoComplete="email" />
          </label>

          <label>
            <span>Password</span>
            <input type="password" name="password" placeholder="Enter password" autoComplete="current-password" />
          </label>

          <div className="nx-login-options">
            <label>
              <input type="checkbox" name="remember" />
              <span>Remember device</span>
            </label>
            <a href="/contact">Need access?</a>
          </div>

          <button type="button">Login</button>
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
