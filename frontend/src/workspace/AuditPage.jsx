import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { listAuditRecords, verifyAuditChain } from "../services/imatchApi";

const ACTION_LABEL = {
  "auth.login": "Signed in",
  "auth.login_failed": "Failed sign-in",
  "auth.logout": "Signed out",
  "biometric.search": "Face search",
  "biometric.verify": "1:1 comparison",
  "biometric.enrol": "Subject enrolled",
  "biometric.template_delete": "Template deleted",
  "biometric.subject_delete": "Subject erased",
  "case.adjudicate": "Candidate adjudicated",
  "case.create": "Case opened",
  "case.update": "Case updated",
  "case.export": "Report exported",
  "admin.user_create": "User created",
  "admin.api_key_create": "API key issued",
  "admin.api_key_revoke": "API key revoked",
};

const SENSITIVE_ACTIONS = new Set([
  "biometric.search",
  "biometric.verify",
  "biometric.enrol",
  "biometric.subject_delete",
]);

export function AuditPage() {
  const { hasRole } = useAuth();
  const [records, setRecords] = useState([]);
  const [filter, setFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [verification, setVerification] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setRecords(await listAuditRecords({ action: filter || undefined, limit: 200 }));
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  async function runVerification() {
    setError("");
    try {
      setVerification(await verifyAuditChain());
    } catch (verifyError) {
      setError(verifyError.message);
    }
  }

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Audit trail</h1>
          <p>
            Every consequential action, hash-chained so that editing or deleting a record breaks
            verification from that point onward.
          </p>
        </div>
        {hasRole("admin") && (
          <button type="button" className="wk-button ghost" onClick={runVerification}>
            Verify chain integrity
          </button>
        )}
      </div>

      {error && <div className="wk-error">{error}</div>}

      {verification && (
        <div className={`wk-banner ${verification.valid ? "info" : "critical"}`}>
          <div>
            <strong>
              {verification.valid ? "Chain intact" : "CHAIN BROKEN — treat as a security incident"}
            </strong>
            {verification.valid
              ? `${verification.records_checked} records verified. No record has been altered or removed since it was written.`
              : `Verification failed at record ${verification.broken_at} after ${verification.records_checked} valid records. ${verification.reason}`}
          </div>
        </div>
      )}

      <section className="wk-card">
        <label className="wk-field" style={{ maxWidth: 340 }}>
          <span>Filter by action</span>
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="">All actions</option>
            {Object.entries(ACTION_LABEL).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>

        {loading ? (
          <div className="wk-loading">Loading audit records…</div>
        ) : records.length === 0 ? (
          <div className="wk-empty">No audit records match this filter.</div>
        ) : (
          <div className="wk-table-wrap">
            <table className="wk-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Actor</th>
                  <th>Action</th>
                  <th>Outcome</th>
                  <th>Lawful basis</th>
                  <th>Entry hash</th>
                </tr>
              </thead>
              <tbody>
                {records.map((record) => (
                  <tr key={record.id}>
                    <td className="wk-mono">{new Date(record.created_at).toLocaleString()}</td>
                    <td>
                      {record.actor_label || record.actor_id || "—"}
                      {record.ip_address && (
                        <div className="wk-mono" style={{ color: "var(--muted)", fontSize: 12 }}>
                          {record.ip_address}
                        </div>
                      )}
                    </td>
                    <td>
                      {ACTION_LABEL[record.action] || record.action}
                      {SENSITIVE_ACTIONS.has(record.action) && (
                        <span className="wk-chip neutral" style={{ marginLeft: 8 }}>
                          biometric
                        </span>
                      )}
                    </td>
                    <td>{record.outcome}</td>
                    <td style={{ maxWidth: 260 }}>{record.lawful_basis || "—"}</td>
                    <td className="wk-mono" title={record.entry_hash}>
                      {record.entry_hash.slice(0, 12)}…
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="wk-notice">
          The chain proves records have not been edited since they were written. It does not
          prove the log is complete — someone with database access could still truncate the most
          recent entries. Ship the mirrored JSONL to write-once storage if you need that
          guarantee too.
        </p>
      </section>
    </>
  );
}
