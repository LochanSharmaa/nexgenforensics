import { useCallback, useEffect, useState } from "react";
import { deleteSubject, enrolSubject, listCases, listSubjects } from "../services/imatchApi";
import { ImageDropZone } from "./components/ImageDropZone";
import { ProbeReport } from "./components/ProbeReport";

export function EnrolPage() {
  const [cases, setCases] = useState([]);
  const [subjects, setSubjects] = useState([]);
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({
    displayName: "",
    externalRef: "",
    notes: "",
    caseId: "",
    subjectId: "",
    lawfulBasis: "",
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const loadSubjects = useCallback(async () => {
    try {
      setSubjects(await listSubjects());
    } catch {
      setSubjects([]);
    }
  }, []);

  useEffect(() => {
    listCases().then(setCases).catch(() => setCases([]));
    loadSubjects();
  }, [loadSubjects]);

  async function handleSubmit(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const response = await enrolSubject({ file, ...form });
      setResult(response);
      setFile(null);
      setForm({ displayName: "", externalRef: "", notes: "", caseId: "", subjectId: "", lawfulBasis: "" });
      await loadSubjects();
    } catch (enrolError) {
      setError(
        enrolError.payload?.detail?.message
          ? `${enrolError.payload.detail.message} (quality ${enrolError.payload.detail.quality?.score})`
          : enrolError.message,
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete(subject) {
    const label = subject.display_name || subject.external_ref || subject.id;
    if (!window.confirm(`Permanently erase ${label}? All templates and enrolment images are deleted.`)) {
      return;
    }
    setError("");
    try {
      await deleteSubject(subject.id);
      await loadSubjects();
    } catch (deleteError) {
      setError(deleteError.message);
    }
  }

  return (
    <>
      <div className="wk-page-head">
        <div>
          <h1>Enrol a subject</h1>
          <p>
            Enrolment determines who this system is capable of finding. Only enrol people your
            organisation has a lawful basis to hold biometric data on.
          </p>
        </div>
      </div>

      {error && <div className="wk-error">{error}</div>}
      {result && (
        <div className="wk-success">
          Enrolled {result.subject.display_name || result.subject.id} — template{" "}
          <span className="wk-mono">{result.template.id.slice(0, 12)}</span> at quality{" "}
          {result.quality.score}.
        </div>
      )}

      <div className="wk-grid two">
        <section className="wk-card">
          <h2>Enrolment image</h2>
          <p>
            One face, well lit, close to frontal. A weak enrolment image degrades every future
            search against this subject, so low-quality images are refused rather than accepted
            quietly.
          </p>

          <form onSubmit={handleSubmit}>
            <ImageDropZone
              id="enrol-image"
              label="Drop or select the enrolment image"
              file={file}
              onChange={setFile}
            />

            <div style={{ height: 18 }} />

            <label className="wk-field">
              <span>Display name</span>
              <input
                maxLength={200}
                value={form.displayName}
                onChange={(event) => setForm({ ...form, displayName: event.target.value })}
              />
            </label>

            <label className="wk-field">
              <span>External reference</span>
              <input
                maxLength={120}
                value={form.externalRef}
                onChange={(event) => setForm({ ...form, externalRef: event.target.value })}
                placeholder="Record number in your system of origin"
              />
              <small>Reusing an existing reference adds this image to that subject.</small>
            </label>

            <label className="wk-field">
              <span>Add to an existing subject</span>
              <select
                value={form.subjectId}
                onChange={(event) => setForm({ ...form, subjectId: event.target.value })}
              >
                <option value="">Create a new subject</option>
                {subjects.map((subject) => (
                  <option key={subject.id} value={subject.id}>
                    {subject.display_name || subject.external_ref || subject.id.slice(0, 8)}
                  </option>
                ))}
              </select>
              <small>
                Several images of the same person improve recall across pose and lighting.
              </small>
            </label>

            <label className="wk-field">
              <span>Case</span>
              <select
                value={form.caseId}
                onChange={(event) => setForm({ ...form, caseId: event.target.value })}
              >
                <option value="">No case</option>
                {cases.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.reference} — {item.title}
                  </option>
                ))}
              </select>
            </label>

            <label className="wk-field">
              <span className="wk-required">Lawful basis</span>
              <input
                required
                maxLength={500}
                value={form.lawfulBasis}
                onChange={(event) => setForm({ ...form, lawfulBasis: event.target.value })}
                placeholder="Consent recorded 2026-02-12 / warrant 2026/114"
              />
            </label>

            <label className="wk-field">
              <span>Notes</span>
              <textarea
                maxLength={5000}
                value={form.notes}
                onChange={(event) => setForm({ ...form, notes: event.target.value })}
              />
            </label>

            <button type="submit" className="wk-button" disabled={busy || !file}>
              {busy ? "Enrolling…" : "Enrol subject"}
            </button>
          </form>
        </section>

        <section className="wk-card">
          <h2>Enrolment assessment</h2>
          <p>Quality of the image just enrolled.</p>

          {result ? (
            <ProbeReport
              probe={{
                quality: result.quality,
                liveness: result.liveness,
                deepfake_risk: 0,
                faces_detected: 1,
                detector: result.template.detector,
                pose: { yaw: 0, pitch: 0, roll: 0 },
              }}
              reasons={result.warnings}
            />
          ) : (
            <div className="wk-empty">Enrol an image to see its quality assessment.</div>
          )}
        </section>
      </div>

      <section className="wk-card">
        <h2>Enrolled subjects</h2>
        <p>
          Erasing a subject deletes their templates and enrolment images outright. Past search
          records are retained, because removing them would destroy the audit trail.
        </p>

        {subjects.length === 0 ? (
          <div className="wk-empty">Nobody is enrolled yet. Searches will return no candidates.</div>
        ) : (
          <div className="wk-table-wrap">
            <table className="wk-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>External reference</th>
                  <th>Enrolled</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {subjects.map((subject) => (
                  <tr key={subject.id}>
                    <td>{subject.display_name || "(unnamed)"}</td>
                    <td className="wk-mono">{subject.external_ref || "—"}</td>
                    <td className="wk-mono">{new Date(subject.created_at).toLocaleDateString()}</td>
                    <td>
                      <button
                        type="button"
                        className="wk-button small danger"
                        onClick={() => handleDelete(subject)}
                      >
                        Erase
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </>
  );
}
