import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import {
  forgotPassword,
  register,
  resendOtp,
  resetPassword,
  verifyEmail,
} from "../../services/imatchApi";
import "./LoginPage.css";
import "./AuthFlowPages.css";

/**
 * Registration, e-mail verification and password reset.
 *
 * Reuses LoginPage.css so these screens are visually the same surface as
 * sign-in rather than an imitation of it; AuthFlowPages.css only adds what is
 * genuinely new (the OTP boxes, the strength meter, the resend timer).
 *
 * Every submit path guards against double submission with a `busy` flag. On
 * these endpoints a duplicate click is not merely untidy: a second /register
 * consumes a rate-limit slot, and a second /resend-otp burns one of the three
 * codes the account is allowed in half an hour.
 */

function Shell({ kicker, title, blurb, children, footer }) {
  return (
    <section className="nx-login-page nx-auth-narrow" id="top">
      <div className="nx-login-copy">
        <p className="nx-kicker">{kicker}</p>
        <h1>{title}</h1>
        <p>{blurb}</p>
      </div>
      <div className="nx-login-panel">
        {children}
        {footer && <div className="nx-auth-footer">{footer}</div>}
      </div>
    </section>
  );
}

function Notice({ error, message }) {
  if (error) {
    return <p className="nx-login-error" role="alert">{error}</p>;
  }
  if (message) {
    return <p className="nx-auth-ok" role="status">{message}</p>;
  }
  return null;
}

/** Mirrors the server's rule (12+ chars, 3 of 4 character classes) so the
 *  requirement is visible before submitting, not discovered by rejection. */
function passwordScore(value) {
  if (!value) return { score: 0, label: "", hint: "At least 12 characters." };
  const classes = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((re) => re.test(value)).length;
  const longEnough = value.length >= 12;
  if (!longEnough) return { score: 1, label: "Too short", hint: `${12 - value.length} more characters needed.` };
  if (classes < 3) return { score: 2, label: "Too simple", hint: "Mix upper case, lower case, digits or symbols." };
  if (value.length >= 16 && classes === 4) return { score: 4, label: "Strong", hint: "" };
  return { score: 3, label: "Good", hint: "" };
}

// ------------------------------------------------------------- register --

export function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ fullName: "", email: "", password: "", confirmPassword: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const strength = useMemo(() => passwordScore(form.password), [form.password]);
  const mismatch = form.confirmPassword && form.password !== form.confirmPassword;
  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  async function handleSubmit(event) {
    event.preventDefault();
    if (busy) return;
    if (mismatch) {
      setError("Passwords do not match.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await register(form);
      // The code goes to the mailbox, so the address is carried forward rather
      // than retyped -- a mistyped address here would send the code nowhere.
      navigate("/verify-email", { replace: true, state: { email: form.email } });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell
      kicker="Create Account"
      title="Request iMATCH access"
      blurb="Create your account, confirm your e-mail address, and your workspace is ready. Every biometric operation you perform is recorded against your account."
      footer={<>Already have an account? <Link to="/login">Sign in</Link></>}
    >
      <form className="nx-login-form" onSubmit={handleSubmit} noValidate>
        <Notice error={error} />
        <label>
          <span>Full Name</span>
          <input type="text" required value={form.fullName} onChange={set("fullName")}
                 placeholder="Alex Morgan" autoComplete="name" />
        </label>
        <label>
          <span>Email Address</span>
          <input type="email" required value={form.email} onChange={set("email")}
                 placeholder="analyst@agency.gov" autoComplete="email" />
        </label>
        <label>
          <span>Password</span>
          <input type="password" required value={form.password} onChange={set("password")}
                 placeholder="At least 12 characters" autoComplete="new-password" />
          {form.password && (
            <span className={`nx-strength s${strength.score}`}>
              <i /><i /><i /><i />
              <b>{strength.label}</b>{strength.hint && <em>{strength.hint}</em>}
            </span>
          )}
        </label>
        <label>
          <span>Confirm Password</span>
          <input type="password" required value={form.confirmPassword}
                 onChange={set("confirmPassword")} autoComplete="new-password" />
          {mismatch && <span className="nx-field-error">Passwords do not match.</span>}
        </label>
        <button type="submit" disabled={busy || mismatch}>
          {busy ? "Creating account…" : "Create account"}
        </button>
      </form>
    </Shell>
  );
}

// -------------------------------------------------------- verify e-mail --

export function VerifyEmailPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [email, setEmail] = useState(
    () => window.history.state?.usr?.email || params.get("email") || "",
  );
  const [digits, setDigits] = useState(Array(6).fill(""));
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [cooldown, setCooldown] = useState(0);
  const boxes = useRef([]);

  useEffect(() => {
    if (cooldown <= 0) return undefined;
    const timer = setTimeout(() => setCooldown((value) => value - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  const code = digits.join("");

  function handleDigit(index, raw) {
    const value = raw.replace(/\D/g, "");
    if (!value) {
      setDigits((prev) => prev.map((d, i) => (i === index ? "" : d)));
      return;
    }
    // Handles paste of a whole code into the first box as well as typing.
    setDigits((prev) => {
      const next = [...prev];
      value.split("").forEach((char, offset) => {
        if (index + offset < 6) next[index + offset] = char;
      });
      return next;
    });
    const landing = Math.min(index + value.length, 5);
    boxes.current[landing]?.focus();
  }

  function handleKey(index, event) {
    if (event.key === "Backspace" && !digits[index] && index > 0) {
      boxes.current[index - 1]?.focus();
    }
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (busy || code.length !== 6) return;
    setBusy(true);
    setError("");
    try {
      await verifyEmail({ email, otp: code });
      navigate("/login", {
        replace: true,
        state: { notice: "Email verified. You can now sign in." },
      });
    } catch (err) {
      setError(err.message);
      setDigits(Array(6).fill(""));
      boxes.current[0]?.focus();
    } finally {
      setBusy(false);
    }
  }

  async function handleResend() {
    if (busy || cooldown > 0) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await resendOtp({ email });
      setMessage("A new code is on its way. Any previous code is now invalid.");
      setCooldown(60);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell
      kicker="Verify Email"
      title="Enter your 6-digit code"
      blurb={email
        ? `We sent a code to ${email}. It expires in 10 minutes and can be used once.`
        : "Enter the address you registered with and the code we sent you."}
      footer={<>Wrong address? <Link to="/register">Start again</Link></>}
    >
      <form className="nx-login-form" onSubmit={handleSubmit} noValidate>
        <Notice error={error} message={message} />
        {!email && (
          <label>
            <span>Email Address</span>
            <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                   autoComplete="email" />
          </label>
        )}
        <div className="nx-otp" role="group" aria-label="Verification code">
          {digits.map((digit, index) => (
            <input
              key={index}
              ref={(el) => { boxes.current[index] = el; }}
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={6}
              value={digit}
              aria-label={`Digit ${index + 1}`}
              onChange={(e) => handleDigit(index, e.target.value)}
              onKeyDown={(e) => handleKey(index, e)}
            />
          ))}
        </div>
        <button type="submit" disabled={busy || code.length !== 6}>
          {busy ? "Verifying…" : "Verify email"}
        </button>
        <button type="button" className="nx-linkish" onClick={handleResend}
                disabled={busy || cooldown > 0 || !email}>
          {cooldown > 0 ? `Resend code in ${cooldown}s` : "Send a new code"}
        </button>
      </form>
    </Shell>
  );
}

// ------------------------------------------------------- forgot password --

export function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await forgotPassword({ email });
      // The server deliberately answers the same way whether or not the
      // account exists, and this screen must not undo that by behaving
      // differently -- so its own message is unconditional too.
      setMessage(result?.message || "If an account exists, a password reset email has been sent.");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell
      kicker="Password Reset"
      title="Forgot your password?"
      blurb="Enter your e-mail address and we will send you a link to choose a new password. The link expires in 15 minutes."
      footer={<><Link to="/login">Back to sign in</Link></>}
    >
      <form className="nx-login-form" onSubmit={handleSubmit} noValidate>
        <Notice error={error} message={message} />
        {!message && (
          <>
            <label>
              <span>Email Address</span>
              <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)}
                     placeholder="analyst@agency.gov" autoComplete="email" />
            </label>
            <button type="submit" disabled={busy}>
              {busy ? "Sending…" : "Send reset link"}
            </button>
          </>
        )}
      </form>
    </Shell>
  );
}

// -------------------------------------------------------- reset password --

export function ResetPasswordPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const token = params.get("token") || "";
  const [form, setForm] = useState({ password: "", confirmPassword: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const strength = useMemo(() => passwordScore(form.password), [form.password]);
  const mismatch = form.confirmPassword && form.password !== form.confirmPassword;
  const set = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  async function handleSubmit(event) {
    event.preventDefault();
    if (busy || mismatch) return;
    setBusy(true);
    setError("");
    try {
      await resetPassword({ token, ...form });
      navigate("/login", {
        replace: true,
        state: { notice: "Password updated. You can now sign in." },
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!token) {
    return (
      <Shell kicker="Password Reset" title="This link is incomplete"
             blurb="The reset link is missing its token. Request a new one."
             footer={<Link to="/forgot-password">Request a new link</Link>}>
        <p className="nx-login-error" role="alert">No reset token was supplied.</p>
      </Shell>
    );
  }

  return (
    <Shell
      kicker="Password Reset"
      title="Choose a new password"
      blurb="Once you set a new password you will be signed out everywhere and will need to sign in again."
      footer={<Link to="/login">Back to sign in</Link>}
    >
      <form className="nx-login-form" onSubmit={handleSubmit} noValidate>
        <Notice error={error} />
        <label>
          <span>New Password</span>
          <input type="password" required value={form.password} onChange={set("password")}
                 placeholder="At least 12 characters" autoComplete="new-password" />
          {form.password && (
            <span className={`nx-strength s${strength.score}`}>
              <i /><i /><i /><i />
              <b>{strength.label}</b>{strength.hint && <em>{strength.hint}</em>}
            </span>
          )}
        </label>
        <label>
          <span>Confirm New Password</span>
          <input type="password" required value={form.confirmPassword}
                 onChange={set("confirmPassword")} autoComplete="new-password" />
          {mismatch && <span className="nx-field-error">Passwords do not match.</span>}
        </label>
        <button type="submit" disabled={busy || mismatch}>
          {busy ? "Updating…" : "Update password"}
        </button>
      </form>
    </Shell>
  );
}
