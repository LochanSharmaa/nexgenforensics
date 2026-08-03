const METRICS = [
  ["overall", "Overall quality"],
  ["sharpness", "Sharpness"],
  ["detail", "True detail (band-limit)"],
  ["noise", "Noise (higher = cleaner)"],
  ["contrast", "Contrast"],
  ["brightness", "Brightness"],
  ["resolution", "Resolution"],
  ["compression", "Compression"],
];

/**
 * Before/after quality metrics.
 *
 * Presented as measurements, never as confidence: a sharper image does not make
 * a match more likely to be correct, and this table must not imply that it
 * does. The "true detail" row is the honest one — it is derived from the
 * measured spectral cut-off, so generative upscaling does not move it much,
 * which is exactly the point.
 */
export function MetricDelta({ before, after }) {
  if (!before || !after) return null;

  return (
    <table className="wk-table compact">
      <thead>
        <tr>
          <th>Metric</th>
          <th className="num">Original</th>
          <th className="num">Enhanced</th>
          <th className="num">Δ</th>
        </tr>
      </thead>
      <tbody>
        {METRICS.filter(([key]) => key in before).map(([key, label]) => {
          const from = Number(before[key] ?? 0);
          const to = Number(after[key] ?? 0);
          const delta = to - from;
          const tone = Math.abs(delta) < 0.005 ? "neutral" : delta > 0 ? "good" : "bad";
          return (
            <tr key={key}>
              <td>{label}</td>
              <td className="num">{from.toFixed(3)}</td>
              <td className="num">{to.toFixed(3)}</td>
              <td className={`num wk-delta ${tone}`}>
                {delta >= 0 ? "+" : ""}
                {delta.toFixed(3)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
