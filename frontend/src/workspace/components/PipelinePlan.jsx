/**
 * The processing plan, applied and skipped alike.
 *
 * The skipped stages are not clutter. "Deblur skipped: blur confidence R² 0.31,
 * below the 0.50 trust floor" tells an examiner the system considered the stage
 * and had a measured reason not to run it — which is what makes the pipeline
 * explainable rather than merely logged.
 */
export function PipelinePlan({ plan, stages }) {
  if (!plan?.stages?.length) return null;

  // Prefer execution records (they carry timing and the change check); fall
  // back to the planned entry for stages that were skipped or not yet run.
  const executed = new Map((stages || []).map((stage) => [stage.name, stage]));

  return (
    <div className="wk-plan">
      <table className="wk-table compact">
        <thead>
          <tr>
            <th>Stage</th>
            <th>Track</th>
            <th>Status</th>
            <th>Why</th>
            <th className="num">Time</th>
            <th className="num">VRAM</th>
          </tr>
        </thead>
        <tbody>
          {plan.stages.map((stage) => {
            const run = executed.get(stage.name);
            const applied = stage.selected && run;
            return (
              <tr key={`${stage.task}-${stage.name}`} className={applied ? "" : "wk-plan-skipped"}>
                <td>
                  <code>{stage.name.replace(/^</, "").replace(/>$/, "")}</code>
                  <small className="wk-plan-task">{stage.task}</small>
                </td>
                <td>
                  <span className={`wk-track ${stage.track}`}>
                    {stage.track === "reconstruction" ? "reconstruction" : "measurement"}
                  </span>
                </td>
                <td>
                  {applied ? (
                    run.changed ? (
                      <span className="wk-chip good">applied</span>
                    ) : (
                      <span className="wk-chip review">no change</span>
                    )
                  ) : stage.selected ? (
                    <span className="wk-chip review">not run</span>
                  ) : (
                    <span className="wk-chip neutral">skipped</span>
                  )}
                </td>
                <td className="wk-plan-why">
                  {stage.selected ? stage.rationale : stage.skip_reason}
                  {run?.notes?.length > 0 && (
                    <div className="wk-plan-note">{run.notes.join(" ")}</div>
                  )}
                </td>
                <td className="num">{run ? `${Math.round(run.duration_ms)} ms` : "—"}</td>
                <td className="num">
                  {run && run.vram_peak_mb > 0 ? `${Math.round(run.vram_peak_mb)} MB` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <small className="wk-plan-foot">
        Ruleset v{plan.ruleset_version}. Ordering is fixed by physics: deinterlace, deblock,
        denoise, tone, deblur, then any reconstruction. Every stage is gated on the confidence of
        its own measurement.
      </small>
    </div>
  );
}
