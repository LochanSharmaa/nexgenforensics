import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { createCase, listCases } from "../services/imatchApi";

const STATUS_TONE = {
  open: "good",
  pending_review: "review",
  closed: "neutral",
  archived: "neutral",
};

export function CaseListPage() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ reference: "", title: "", description: "", lawfulBasis: "" });

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setCases(await listCases());
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function handleCreate(event) {
    event.preventDefault();
    setCreating(true);
    setError("");
    try {
      await createCase(form);
      setForm({ reference: "", title: "", description: "", lawfulBasis: "" });
      await load();
    } catch (createError) {
      setError(createError.message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Cases</h1>
          <p>
            Every biometric search is attached to a case, so the reason a person was searched
            for stays attached to the result.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}

      <div className="wk-grid two">
        <section className="wk-card">
          <h2>Open cases</h2>
          <p>Investigators see their own cases; supervisors see the whole tenant.</p>

          {loading ? (
            <div className="wk-loading">Loading cases…</div>
          ) : cases.length === 0 ? (
            <div className="wk-empty">No cases yet. Create one to start searching.</div>
          ) : (
            <div className="wk-table-wrap">
              <table className="wk-table">
                <thead>
                  <tr>
                    <th>Reference</th>
                    <th>Title</th>
                    <th>Status</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((item) => (
                    <tr key={item.id}>
                      <td className="wk-mono">
                        <Link to={`/workspace/cases/${item.id}`}>{item.reference}</Link>
                      </td>
                      <td>{item.title}</td>
                      <td>
                        <span className={`wk-chip ${STATUS_TONE[item.status] || "neutral"}`}>
                          {item.status.replace("_", " ")}
                        </span>
                      </td>
                      <td className="wk-mono">{new Date(item.updated_at).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="wk-card">
          <h2>Open a new case</h2>
          <p>The lawful basis recorded here is inherited by every search run under this case.</p>

          <form onSubmit={handleCreate}>
            <label className="wk-field">
              <span className="wk-required">Case reference</span>
              <input
                required
                maxLength={120}
                value={form.reference}
                onChange={(event) => setForm({ ...form, reference: event.target.value })}
                placeholder="OP-2026-0114"
              />
            </label>

            <label className="wk-field">
              <span className="wk-required">Title</span>
              <input
                required
                maxLength={250}
                value={form.title}
                onChange={(event) => setForm({ ...form, title: event.target.value })}
                placeholder="Commercial burglary series, north district"
              />
            </label>

            <label className="wk-field">
              <span>Lawful basis</span>
              <input
                maxLength={500}
                value={form.lawfulBasis}
                onChange={(event) => setForm({ ...form, lawfulBasis: event.target.value })}
                placeholder="Warrant 2026/114, issued 12 Feb 2026"
              />
              <small>
                Cite the authority permitting biometric processing on this case. It is written
                into the audit chain and reproduced in the exported report.
              </small>
            </label>

            <label className="wk-field">
              <span>Description</span>
              <textarea
                maxLength={5000}
                value={form.description}
                onChange={(event) => setForm({ ...form, description: event.target.value })}
              />
            </label>

            <button type="submit" className="wk-button" disabled={creating}>
              {creating ? "Creating…" : "Create case"}
            </button>
          </form>
        </section>
      </div>
    </>
  );
}
