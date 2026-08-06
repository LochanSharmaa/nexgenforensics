import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { fetchAuditImage, listAuditRecords, verifyAuditChain } from "../services/imatchApi";

const ACTION_LABEL = {
  "auth.login": "Signed in",
  "auth.login_failed": "Failed sign-in",
  "auth.logout": "Signed out",
  "biometric.search": "Face search",
  "biometric.verify": "1:1 comparison",
  "biometric.enrol": "Subject enrolled",
  "biometric.template_delete": "Template deleted",
  "biometric.subject_delete": "Subject erased",
  "evidence.enhance": "Evidence enhanced",
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

function parseDetail(detail) {
  try {
    const parsed = JSON.parse(detail || "{}");
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

/**
 * One referenced image as a thumbnail. The bytes are only fetched once the
 * row scrolls near the viewport — an audit page can hold 200 rows, and
 * loading every image up front would hammer the API for rows never seen.
 */
function AuditThumb({ recordId, image }) {
  const holderRef = useRef(null);
  const urlRef = useRef(null);
  const [visible, setVisible] = useState(false);
  const [src, setSrc] = useState("");
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    const node = holderRef.current;
    if (!node) return undefined;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible || src || missing) return undefined;
    let alive = true;
    fetchAuditImage(recordId, image.key)
      .then((objectUrl) => {
        if (!alive) {
          URL.revokeObjectURL(objectUrl);
          return;
        }
        urlRef.current = objectUrl;
        setSrc(objectUrl);
      })
      .catch(() => {
        if (alive) setMissing(true);
      });
    return () => {
      alive = false;
    };
  }, [visible, src, missing, recordId, image.key]);

  // Object URLs leak until revoked; release on unmount only, never on rerender.
  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  const sha = image.sha256 ? ` — SHA-256 ${image.sha256}` : "";

  return (
    <figure className="wk-evidence-item" ref={holderRef}>
      {src ? (
        <img
          className="wk-evidence-thumb"
          src={src}
          alt={image.label}
          title={`${image.label}${sha}. Click to view full size.`}
          onClick={() => window.open(src, "_blank", "noopener")}
        />
      ) : missing ? (
        <span
          className="wk-evidence-thumb wk-evidence-missing"
          title={`${image.label}${sha}. The stored bytes have passed their retention window; the hash in the chained entry still identifies them.`}
        >
          gone
        </span>
      ) : (
        <span className="wk-evidence-thumb wk-evidence-pending" aria-hidden="true" />
      )}
      <figcaption>{image.label}</figcaption>
    </figure>
  );
}

/** The images a row's action compared, with the comparison relation spelled out. */
function AuditImages({ record }) {
  if (!record.images || record.images.length === 0) {
    return <span className="wk-evidence-none">—</span>;
  }

  const detail = parseDetail(record.detail);
  const separator =
    record.action === "biometric.verify" ? "⇌" : record.images.length === 2 ? "→" : null;

  let note = "";
  if (record.action === "biometric.verify" && typeof detail.similarity === "number") {
    note = `similarity ${detail.similarity.toFixed(3)}`;
  } else if (record.action === "biometric.search" && typeof detail.gallery_size === "number") {
    note = `vs gallery of ${detail.gallery_size}`;
  }

  return (
    <div className="wk-evidence">
      <div className="wk-evidence-row">
        {record.images.map((image, index) => (
          <span className="wk-evidence-cell" key={image.key}>
            {index > 0 && separator && (
              <span className="wk-evidence-sep" aria-hidden="true">
                {separator}
              </span>
            )}
            <AuditThumb recordId={record.id} image={image} />
          </span>
        ))}
      </div>
      {note && <div className="wk-evidence-note">{note}</div>}
    </div>
  );
}

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
            verification from that point onward — including the exact images each comparison ran
            on.
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
                  <th>Images compared</th>
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
                    <td>
                      <AuditImages record={record} />
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
          guarantee too. Image bytes are kept for the probe retention window; the SHA-256 in
          each entry identifies them permanently either way.
        </p>
      </section>
    </>
  );
}
