import { useState } from "react";
import { adjudicateCandidate } from "../../services/imatchApi";

const ADJUDICATION_LABEL = {
  pending: ["Awaiting review", "neutral"],
  confirmed: ["Confirmed by examiner", "good"],
  eliminated: ["Eliminated", "bad"],
  inconclusive: ["Inconclusive", "review"],
};

function bandFor(score, thresholds) {
  if (!thresholds) return "";
  if (score >= thresholds.match) return "";
  if (score >= thresholds.review) return "review";
  return "low";
}

/**
 * Ranked candidates with examiner adjudication.
 *
 * The similarity column is shown as a raw cosine value, not a percentage.
 * Rendering 0.62 as "62% confident" invites the reader to treat it as a
 * probability that the two images are the same person, which it is not.
 */
export function CandidateTable({ candidates, thresholds, onUpdated, readOnly = false }) {
  const [busyId, setBusyId] = useState(null);
  const [error, setError] = useState("");
  const [notes, setNotes] = useState({});

  if (!candidates?.length) {
    return (
      <div className="wk-empty">
        No candidate cleared the minimum similarity. Either nobody in the gallery resembles this
        probe, or the person is not enrolled.
      </div>
    );
  }

  async function adjudicate(candidate, verdict) {
    setBusyId(candidate.id);
    setError("");
    try {
      await adjudicateCandidate(candidate.id, {
        adjudication: verdict,
        notes: notes[candidate.id] || "",
      });
      onUpdated?.();
    } catch (adjudicationError) {
      setError(adjudicationError.message);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <>
      {error && <div className="wk-error">{error}</div>}

      <div className="wk-table-wrap">
        <table className="wk-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Subject</th>
              <th>Similarity</th>
              <th>Std. dev. from field</th>
              <th>Examiner verdict</th>
              {!readOnly && <th>Adjudicate</th>}
            </tr>
          </thead>
          <tbody>
            {candidates.map((candidate) => {
              const [label, tone] = ADJUDICATION_LABEL[candidate.adjudication] || ADJUDICATION_LABEL.pending;
              return (
                <tr key={candidate.id}>
                  <td>{candidate.rank}</td>
                  <td>
                    <strong>{candidate.subject_name || "(unnamed subject)"}</strong>
                    {candidate.external_ref && (
                      <div className="wk-mono" style={{ color: "var(--muted)" }}>
                        {candidate.external_ref}
                      </div>
                    )}
                  </td>
                  <td>
                    <span className="wk-mono">{candidate.score.toFixed(4)}</span>
                    <div className="wk-score-bar">
                      <i
                        className={bandFor(candidate.score, thresholds)}
                        style={{ width: `${Math.max(0, Math.min(100, candidate.score * 100))}%` }}
                      />
                    </div>
                  </td>
                  <td className="wk-mono">
                    {candidate.normalized_score ? candidate.normalized_score.toFixed(2) : "—"}
                  </td>
                  <td>
                    <span className={`wk-chip ${tone}`}>{label}</span>
                    {candidate.examiner_notes && (
                      <div style={{ marginTop: 6, fontSize: 13, color: "var(--muted)" }}>
                        {candidate.examiner_notes}
                      </div>
                    )}
                  </td>
                  {!readOnly && (
                    <td>
                      <input
                        type="text"
                        placeholder="Reasoning (recorded)"
                        value={notes[candidate.id] || ""}
                        onChange={(event) =>
                          setNotes((current) => ({ ...current, [candidate.id]: event.target.value }))
                        }
                        style={{
                          width: "100%",
                          marginBottom: 8,
                          padding: "7px 10px",
                          border: "1px solid var(--line)",
                          borderRadius: 7,
                          font: "inherit",
                          fontSize: 13,
                        }}
                      />
                      <div className="wk-actions">
                        <button
                          type="button"
                          className="wk-button small"
                          disabled={busyId === candidate.id}
                          onClick={() => adjudicate(candidate, "confirmed")}
                        >
                          Confirm
                        </button>
                        <button
                          type="button"
                          className="wk-button small ghost"
                          disabled={busyId === candidate.id}
                          onClick={() => adjudicate(candidate, "eliminated")}
                        >
                          Eliminate
                        </button>
                        <button
                          type="button"
                          className="wk-button small ghost"
                          disabled={busyId === candidate.id}
                          onClick={() => adjudicate(candidate, "inconclusive")}
                        >
                          Inconclusive
                        </button>
                      </div>
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="wk-notice">
        Confirming a candidate records <em>your</em> professional judgement, not the system&rsquo;s.
        The engine never sets this field. Verify the images side by side before confirming, and
        state your reasoning — both are written to the audit record and appear in the case report.
      </p>
    </>
  );
}
