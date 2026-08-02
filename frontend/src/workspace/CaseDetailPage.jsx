import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  fetchCaseReport,
  getCase,
  listCandidates,
  listSearches,
  updateCase,
} from "../services/imatchApi";
import { CandidateTable } from "./components/CandidateTable";

const DECISION_TONE = {
  candidate_match: "good",
  review_required: "review",
  no_match: "neutral",
  probe_rejected: "bad",
  recognition_unavailable: "bad",
};

export function CaseDetailPage() {
  const { caseId } = useParams();
  const [caseRecord, setCaseRecord] = useState(null);
  const [searches, setSearches] = useState([]);
  const [expanded, setExpanded] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [record, runs] = await Promise.all([getCase(caseId), listSearches(caseId)]);
      setCaseRecord(record);
      setSearches(runs);
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    load();
  }, [load]);

  async function toggleSearch(searchId) {
    if (expanded === searchId) {
      setExpanded(null);
      setCandidates([]);
      return;
    }
    setExpanded(searchId);
    try {
      setCandidates(await listCandidates(searchId));
    } catch (candidateError) {
      setError(candidateError.message);
    }
  }

  async function changeStatus(status) {
    try {
      setCaseRecord(await updateCase(caseId, { status }));
    } catch (statusError) {
      setError(statusError.message);
    }
  }

  async function downloadReport(format) {
    setError("");
    try {
      const report = await fetchCaseReport(caseId, format);
      // PDF arrives as a Blob already; the text formats are wrapped into one.
      const blob =
        format === "pdf"
          ? report
          : new Blob([format === "markdown" ? report : JSON.stringify(report, null, 2)], {
              type: format === "markdown" ? "text/markdown" : "application/json",
            });
      const extension = { pdf: "pdf", markdown: "md", json: "json" }[format];
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `case-${caseRecord?.reference || caseId}.${extension}`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (reportError) {
      setError(reportError.message);
    }
  }

  if (loading) return <div className="wk-loading">Loading case…</div>;
  if (error && !caseRecord) return <div className="wk-error">{error}</div>;
  if (!caseRecord) return null;

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>{caseRecord.reference}</h1>
          <p>{caseRecord.title}</p>
        </div>
        <div className="wk-actions">
          <Link className="wk-button ghost" to={`/workspace/search?case=${caseId}`}>
            New search
          </Link>
          <button type="button" className="wk-button ghost" onClick={() => downloadReport("pdf")}>
            Export PDF
          </button>
          <button type="button" className="wk-button ghost" onClick={() => downloadReport("markdown")}>
            Export Markdown
          </button>
          <button type="button" className="wk-button ghost" onClick={() => downloadReport("json")}>
            Export JSON
          </button>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <section className="wk-card">
        <div className="wk-metrics">
          <div className="wk-metric">
            <span>Status</span>
            <strong style={{ fontSize: 17 }}>{caseRecord.status.replace("_", " ")}</strong>
          </div>
          <div className="wk-metric">
            <span>Searches run</span>
            <strong>{caseRecord.search_count}</strong>
          </div>
          <div className="wk-metric">
            <span>Subjects enrolled</span>
            <strong>{caseRecord.subject_count}</strong>
          </div>
          <div className="wk-metric">
            <span>Examiner-confirmed</span>
            <strong>{caseRecord.confirmed_count}</strong>
            <small>Human verdicts only</small>
          </div>
        </div>

        {caseRecord.lawful_basis && (
          <p className="wk-notice">Lawful basis on record: {caseRecord.lawful_basis}</p>
        )}
        {caseRecord.description && (
          <p style={{ fontSize: 14.5, lineHeight: 1.65 }}>{caseRecord.description}</p>
        )}

        <div className="wk-actions" style={{ marginTop: 12 }}>
          {caseRecord.status !== "closed" && (
            <button type="button" className="wk-button ghost small" onClick={() => changeStatus("closed")}>
              Close case
            </button>
          )}
          {caseRecord.status === "closed" && (
            <button type="button" className="wk-button ghost small" onClick={() => changeStatus("open")}>
              Reopen case
            </button>
          )}
        </div>
      </section>

      <section className="wk-card">
        <h2>Search history</h2>
        <p>
          Every search on this case, including those that returned nothing. Select one to review
          and adjudicate its candidates.
        </p>

        {searches.length === 0 ? (
          <div className="wk-empty">No searches have been run on this case yet.</div>
        ) : (
          <div className="wk-table-wrap">
            <table className="wk-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Decision</th>
                  <th>Top score</th>
                  <th>Candidates</th>
                  <th>Probe quality</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {searches.map((run) => (
                  <tr key={run.id}>
                    <td className="wk-mono">{new Date(run.created_at).toLocaleString()}</td>
                    <td>
                      <span className={`wk-chip ${DECISION_TONE[run.decision] || "neutral"}`}>
                        {run.decision.replace(/_/g, " ")}
                      </span>
                      {!run.recognition_capable && (
                        <div style={{ marginTop: 5, fontSize: 12, color: "var(--burgundy)" }}>
                          Ran without a recognition model
                        </div>
                      )}
                    </td>
                    <td className="wk-mono">{run.top_score.toFixed(4)}</td>
                    <td>{run.candidate_count}</td>
                    <td className="wk-mono">{run.quality_score.toFixed(3)}</td>
                    <td>
                      <button
                        type="button"
                        className="wk-button small ghost"
                        onClick={() => toggleSearch(run.id)}
                      >
                        {expanded === run.id ? "Hide" : "Review"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {expanded && (
        <section className="wk-card">
          <h2>Candidates</h2>
          <p>
            Search <span className="wk-mono">{expanded.slice(0, 12)}</span>
          </p>
          <CandidateTable
            candidates={candidates}
            thresholds={null}
            onUpdated={async () => setCandidates(await listCandidates(expanded))}
          />
        </section>
      )}
    </>
  );
}
