import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import { destinationFor, setWorkspaceMode } from "../../context/workspaceMode";
import "./ChooseRolePage.css";

/**
 * One-time "how do you want to use iMATCH?" screen, shown after a successful
 * sign-in when no preference is stored on this device.
 *
 * This chooses a DESTINATION, not a permission level — see workspaceMode.js.
 * The wording avoids implying otherwise: neither option is described as giving
 * more or less access, because neither does.
 */

const OPTIONS = [
  {
    id: "investigator",
    title: "Investigator",
    tagline: "Full case workspace",
    blurb:
      "Case management, 1:N gallery search, batch processing, enrolment and the hash-chained audit trail.",
    points: ["Cases and search history", "1:N identification", "Batch processing", "Audit trail"],
  },
  {
    id: "individual",
    title: "Individual",
    tagline: "Single comparison only",
    blurb:
      "A single screen for comparing two photographs. No cases, no batches, nothing else to configure.",
    points: ["Upload two images", "Run one comparison", "See the similarity result"],
  },
];

export function ChooseRolePage() {
  // Rendered without requiring a session: picking a destination grants nothing,
  // and the destinations themselves are still gated. Bouncing to /login from
  // here only got in the way.
  const { isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (loading) {
    return <section className="nx-role-page"><p className="nx-role-loading">Restoring session…</p></section>;
  }

  const from = location.state?.from;

  function choose(mode) {
    setWorkspaceMode(mode);
    navigate(destinationFor(mode, from), { replace: true });
  }

  return (
    <section className="nx-role-page" id="top">
      <div className="nx-role-inner">
        <p className="nx-kicker">{isAuthenticated ? "Signed in" : "iMATCH"}</p>
        <h1>How will you be using iMATCH?</h1>
        <p className="nx-role-sub">
          This sets where you land when you sign in. It does not change your permissions — those
          come from your account and are enforced by the server. You can change this later.
        </p>

        <div className="nx-role-grid">
          {OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              className="nx-role-card"
              data-role={option.id}
              onClick={() => choose(option.id)}
            >
              <span className="nx-role-tagline">{option.tagline}</span>
              <strong className="nx-role-title">{option.title}</strong>
              <span className="nx-role-blurb">{option.blurb}</span>
              <ul className="nx-role-points">
                {option.points.map((point) => (
                  <li key={point}>{point}</li>
                ))}
              </ul>
              <span className="nx-role-cta">Continue as {option.title} →</span>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}
