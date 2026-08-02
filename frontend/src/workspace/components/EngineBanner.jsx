import { useEffect, useState } from "react";
import { fetchEngineStatus } from "../../services/imatchApi";

/**
 * Tells the operator what the engine actually is right now.
 *
 * When the recognition model is missing the service still answers every
 * request, and the scores look entirely normal -- they are just meaningless.
 * That is the most dangerous state this system can be in, so it is surfaced
 * permanently and prominently rather than buried in a status endpoint.
 */
export function EngineBanner({ onStatus }) {
  const [status, setStatus] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchEngineStatus()
      .then((value) => {
        if (cancelled) return;
        setStatus(value);
        onStatus?.(value);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [onStatus]);

  if (failed) {
    return (
      <div className="wk-banner critical" role="alert">
        <div>
          <strong>Cannot reach the recognition service</strong>
          The API did not respond. Searches will fail until the backend on port 8443 is running.
        </div>
      </div>
    );
  }

  // The service cannot start without real weights, so there is no "running
  // without a model" state to warn about any more — that failure now happens at
  // startup rather than silently at search time. A healthy engine needs no
  // banner; the details remain available via onStatus and the status endpoint.
  return null;
}
