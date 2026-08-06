import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  listExaminationExhibits,
  listExaminationNotes,
  saveExaminationNotes,
} from "../../services/imatchApi";

/**
 * The examiner's own sections of the photograph examination report.
 *
 * This panel exists because the report will not write these. The inventory,
 * the measurements and the plates are derived from the case; the question put,
 * the purpose, the observations and the result are opinion, and a system that
 * composed them would be fabricating expert evidence. Anything left blank here
 * prints as "not recorded" — which is a finding a reader can act on, where
 * generated prose would not be.
 *
 * ON THE MARK PICKER
 * ------------------
 * An observation cites the exhibits it is about, and the report builds that
 * observation's comparison grid from the citation. Marks are therefore chosen
 * from the case's actual inventory rather than typed: a typed "S-3" that does
 * not exist reaches the PDF as a printed warning, and catching it here is
 * better than catching it in a document already sent out.
 *
 * A mark that was cited earlier and is no longer in the inventory is still
 * shown, flagged, and removable. Hiding it would silently drop a citation the
 * examiner made.
 */

const SINGLE_SECTIONS = [
  {
    key: "receipt",
    label: "Receipt of material",
    hint: "How and from whom the photographs arrived, and when.",
    rows: 3,
  },
  {
    key: "question",
    label: "Question put for examination",
    hint: "The question you were asked to answer.",
    rows: 3,
  },
  {
    key: "purpose",
    label: "Purpose of examination",
    hint: "What this examination set out to establish.",
    rows: 3,
  },
];

const EMPTY = { receipt: "", question: "", purpose: "", result: "" };

let observationSeq = 0;
const nextKey = () => `obs-${(observationSeq += 1)}`;

function toObservation(note) {
  return { key: nextKey(), body: note?.body || "", marks: note?.marks || [] };
}

export function ExaminationNotesPanel({ caseId }) {
  const [singles, setSingles] = useState(EMPTY);
  const [observations, setObservations] = useState([]);
  const [exhibits, setExhibits] = useState([]);
  const [exhibitsState, setExhibitsState] = useState("loading");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [savedAt, setSavedAt] = useState("");

  // The saved snapshot, so "unsaved changes" reflects what the server holds
  // rather than merely whether the examiner has typed.
  const baseline = useRef("");

  const snapshot = useMemo(
    () => JSON.stringify({ singles, observations: observations.map(({ body, marks }) => ({ body, marks })) }),
    [singles, observations],
  );
  const dirty = !loading && snapshot !== baseline.current;

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const notes = await listExaminationNotes(caseId);
      const next = { ...EMPTY };
      for (const key of Object.keys(EMPTY)) {
        const found = notes.filter((note) => note.section === key);
        next[key] = found.map((note) => note.body).join("\n\n");
      }
      const loadedObservations = notes
        .filter((note) => note.section === "observation")
        .sort((a, b) => a.ordinal - b.ordinal)
        .map(toObservation);

      setSingles(next);
      setObservations(loadedObservations);
      baseline.current = JSON.stringify({
        singles: next,
        observations: loadedObservations.map(({ body, marks }) => ({ body, marks })),
      });
    } catch (loadError) {
      setError(loadError.message);
    } finally {
      setLoading(false);
    }
  }, [caseId]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    setExhibitsState("loading");
    listExaminationExhibits(caseId)
      .then((rows) => {
        if (cancelled) return;
        setExhibits(rows);
        setExhibitsState("ready");
      })
      .catch(() => {
        // A picker that cannot load must not block writing an observation, so
        // this degrades to free entry rather than to an error.
        if (!cancelled) setExhibitsState("failed");
      });
    return () => {
      cancelled = true;
    };
  }, [caseId]);

  // Leaving with unsaved findings loses work the examiner cannot regenerate.
  useEffect(() => {
    if (!dirty) return undefined;
    const warn = (event) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const knownMarks = useMemo(() => new Set(exhibits.map((item) => item.mark)), [exhibits]);

  function setSingle(key, value) {
    setSingles((current) => ({ ...current, [key]: value }));
  }

  function updateObservation(key, patch) {
    setObservations((current) =>
      current.map((item) => (item.key === key ? { ...item, ...patch } : item)),
    );
  }

  function toggleMark(key, mark) {
    setObservations((current) =>
      current.map((item) =>
        item.key === key
          ? {
              ...item,
              marks: item.marks.includes(mark)
                ? item.marks.filter((m) => m !== mark)
                : [...item.marks, mark],
            }
          : item,
      ),
    );
  }

  function moveObservation(index, delta) {
    setObservations((current) => {
      const target = index + delta;
      if (target < 0 || target >= current.length) return current;
      const next = [...current];
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  async function save() {
    setSaving(true);
    setError("");
    try {
      // Assembled in the order the report prints: the single sections, then the
      // observations numbered by their position in this list.
      const payload = [];
      for (const key of ["receipt", "question", "purpose"]) {
        if (singles[key].trim()) payload.push({ section: key, ordinal: 0, body: singles[key], marks: [] });
      }
      observations.forEach((item, index) => {
        if (item.body.trim()) {
          payload.push({
            section: "observation",
            ordinal: index + 1,
            body: item.body,
            marks: item.marks,
          });
        }
      });
      if (singles.result.trim()) {
        payload.push({ section: "result", ordinal: 0, body: singles.result, marks: [] });
      }

      await saveExaminationNotes(caseId, payload);
      baseline.current = snapshot;
      setSavedAt(new Date().toLocaleTimeString());
    } catch (saveError) {
      setError(saveError.message);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <section className="wk-card">
        <h2>Examination findings</h2>
        <div className="wk-loading">Loading findings…</div>
      </section>
    );
  }

  return (
    <section className="wk-card">
      <h2>Examination findings</h2>
      <p>
        Your words, printed in the examination report exactly as written. Anything left blank
        prints as &ldquo;not recorded&rdquo; — nothing here is written for you.
      </p>

      {error && <div className="wk-error">{error}</div>}

      {SINGLE_SECTIONS.map((section) => (
        <label className="wk-field" key={section.key} style={{ marginBottom: 16 }}>
          <span>{section.label}</span>
          <textarea
            rows={section.rows}
            maxLength={20000}
            value={singles[section.key]}
            onChange={(event) => setSingle(section.key, event.target.value)}
          />
          <small>{section.hint}</small>
        </label>
      ))}

      <h3 style={{ fontSize: 15, margin: "22px 0 6px" }}>Secondary identification</h3>
      <p style={{ marginTop: 0 }}>
        Feature-by-feature findings. Each is numbered in the report and printed above a grid of
        the exhibits it cites.
      </p>

      {observations.length === 0 && (
        <div className="wk-empty">No observations recorded.</div>
      )}

      {observations.map((item, index) => {
        const stale = item.marks.filter((mark) => !knownMarks.has(mark));
        return (
          <div
            key={item.key}
            style={{
              border: "1px solid var(--line)",
              borderRadius: 9,
              padding: 14,
              marginBottom: 14,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 8,
              }}
            >
              <strong style={{ fontSize: 14 }}>Observation {index + 1}</strong>
              <div className="wk-actions">
                <button
                  type="button"
                  className="wk-button small ghost"
                  disabled={index === 0}
                  onClick={() => moveObservation(index, -1)}
                  aria-label={`Move observation ${index + 1} up`}
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="wk-button small ghost"
                  disabled={index === observations.length - 1}
                  onClick={() => moveObservation(index, 1)}
                  aria-label={`Move observation ${index + 1} down`}
                >
                  ↓
                </button>
                <button
                  type="button"
                  className="wk-button small ghost"
                  onClick={() =>
                    setObservations((current) => current.filter((row) => row.key !== item.key))
                  }
                >
                  Remove
                </button>
              </div>
            </div>

            <label className="wk-field">
              <span className="wk-required">Finding</span>
              <textarea
                rows={3}
                maxLength={20000}
                value={item.body}
                placeholder="The hairline in Q-1 shows the same asymmetric peak seen in S-1 and S-3."
                onChange={(event) => updateObservation(item.key, { body: event.target.value })}
              />
            </label>

            <div style={{ marginTop: 10 }}>
              <span
                style={{
                  display: "block",
                  fontSize: 12.5,
                  fontWeight: 600,
                  letterSpacing: "0.02em",
                  textTransform: "uppercase",
                  color: "var(--muted)",
                  marginBottom: 6,
                }}
              >
                Exhibits cited
              </span>

              {exhibitsState === "loading" && (
                <span style={{ fontSize: 13, color: "var(--muted)" }}>Loading exhibits…</span>
              )}

              {exhibitsState === "ready" && exhibits.length === 0 && (
                <span style={{ fontSize: 13, color: "var(--muted)" }}>
                  No exhibits are marked on this case yet. Add images to the case first.
                </span>
              )}

              {exhibitsState === "failed" && (
                <span style={{ fontSize: 13, color: "var(--burgundy)" }}>
                  The exhibit list could not be loaded, so citations cannot be picked. The finding
                  above still saves.
                </span>
              )}

              <div style={{ display: "flex", flexWrap: "wrap", gap: 7 }}>
                {exhibits.map((exhibit) => {
                  const on = item.marks.includes(exhibit.mark);
                  return (
                    <button
                      key={exhibit.mark}
                      type="button"
                      className={`wk-button small ${on ? "" : "ghost"}`}
                      aria-pressed={on}
                      title={exhibit.filename}
                      onClick={() => toggleMark(item.key, exhibit.mark)}
                    >
                      {exhibit.mark}
                    </button>
                  );
                })}
                {stale.map((mark) => (
                  <button
                    key={mark}
                    type="button"
                    className="wk-button small ghost"
                    style={{ borderColor: "var(--burgundy)", color: "var(--burgundy)" }}
                    title="Cited but not in this case's exhibit list. Click to remove."
                    onClick={() => toggleMark(item.key, mark)}
                  >
                    {mark} ✕
                  </button>
                ))}
              </div>

              {stale.length > 0 && (
                <p className="wk-notice" style={{ marginTop: 8 }}>
                  {stale.join(", ")} {stale.length === 1 ? "is" : "are"} not in this case&rsquo;s
                  exhibit list. The report will print that alongside the finding.
                </p>
              )}
            </div>
          </div>
        );
      })}

      <button
        type="button"
        className="wk-button ghost small"
        onClick={() => setObservations((current) => [...current, toObservation()])}
      >
        Add observation
      </button>

      <label className="wk-field" style={{ marginTop: 22 }}>
        <span>Result of examination</span>
        <textarea
          rows={4}
          maxLength={20000}
          value={singles.result}
          onChange={(event) => setSingle("result", event.target.value)}
        />
        <small>Your conclusion, printed verbatim as the report&rsquo;s result.</small>
      </label>

      <div className="wk-actions" style={{ marginTop: 16, alignItems: "center" }}>
        <button type="button" className="wk-button" disabled={!dirty || saving} onClick={save}>
          {saving ? "Saving…" : "Save findings"}
        </button>
        {dirty && !saving && (
          <span style={{ fontSize: 13, color: "var(--burgundy)" }}>Unsaved changes</span>
        )}
        {!dirty && savedAt && (
          <span style={{ fontSize: 13, color: "var(--muted)" }}>Saved at {savedAt}</span>
        )}
      </div>
    </section>
  );
}
