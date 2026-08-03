/**
 * Image Intelligence Engine (IIE) client — public-web image provenance.
 *
 * WHAT THIS IS, AND WHAT IT IS NOT
 * --------------------------------
 * This service answers one question: **where does this photograph already
 * appear on the public web, and what do those pages say?** It is reverse
 * *image* search plus page reading. It performs no facial recognition: it does
 * not identify people from facial features, does not compare faces, and holds
 * no biometric data. Every comparison it makes is over file content —
 * cryptographic and perceptual hashes.
 *
 * That is deliberately the opposite half of iMATCH's face search. Face search
 * compares a probe against subjects your organisation has lawfully enrolled.
 * Provenance never touches a face; it tracks where a *file* has been published.
 * The two answer different questions and neither substitutes for the other.
 *
 * AUTHENTICATION
 * --------------
 * There is no second sign-in. IIE verifies the same bearer token the workspace
 * already holds, using the shared NEXGEN_JWT_SECRET, and provisions a local
 * account keyed by the investigator's iMATCH subject id so audit entries have a
 * real actor. If IIE_IMATCH_JWT_SECRET is not configured on the IIE side, every
 * call here returns 401 — federation is opt-in, never assumed.
 *
 * ORIGIN
 * ------
 * IIE runs as its own service on its own port, so these calls are cross-origin.
 * That means three things must agree or every request fails:
 *   1. VITE_IIE_API_BASE (below)
 *   2. IIE_CORS_ORIGINS on the IIE service, including this page's origin
 *   3. CSP `connect-src`, wherever this app is served from
 * No cookies are used — the bearer token travels in the Authorization header —
 * so SameSite is not a factor here.
 */

import { tokenStore } from "./imatchApi";

/**
 * Development origin, derived from the page's own hostname.
 *
 * Same reasoning as imatchApi's DEV_BASE: `localhost` and `127.0.0.1` are
 * different sites to a browser, and hardcoding one guarantees a mismatch for
 * anyone using the other.
 */
const DEV_BASE =
  typeof window !== "undefined" && window.location?.hostname
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://127.0.0.1:8000";

const configured = (import.meta.env?.VITE_IIE_API_BASE || "").trim();

/**
 * `same-origin` routes through the Vite dev proxy (`/iie` → :8000), which is
 * how imatch_api is already reached locally. Same-origin means no CORS grant
 * per Vite port and no cross-origin request carrying a bearer token — the
 * failure mode this codebase has been bitten by before.
 */
const SAME_ORIGIN = configured === "same-origin" || configured === "/";

export const IIE_BASE = SAME_ORIGIN
  ? "/iie"
  : configured
    ? configured.replace(/\/+$/, "")
    : import.meta.env?.DEV
      ? DEV_BASE
      : "";

/** Whether provenance is configured at all. Used to hide the feature rather
 *  than let an investigator click into a guaranteed failure. */
export const provenanceEnabled = Boolean(IIE_BASE);

const API = `${IIE_BASE}/api/v1`;

/**
 * IIE returns RFC 9457 problem documents. Surfacing `detail` verbatim matters:
 * a blocked workflow transition explains *every* failed precondition at once,
 * and truncating that to a generic message would send the investigator round
 * the loop fixing one blocker at a time.
 */
async function request(path, { method = "GET", body, headers = {} } = {}) {
  const token = tokenStore.access;
  const requestHeaders = { ...headers };
  if (token) requestHeaders.Authorization = `Bearer ${token}`;
  if (body && !(body instanceof FormData)) {
    requestHeaders["Content-Type"] = "application/json";
  }

  let response;
  try {
    response = await fetch(`${API}${path}`, {
      method,
      headers: requestHeaders,
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    });
  } catch (networkError) {
    throw new Error(
      `Could not reach the provenance service at ${IIE_BASE}. ` +
        `Check it is running and that this origin is in IIE_CORS_ORIGINS. (${networkError.message})`,
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const problem = await response.json();
      detail = problem.detail || problem.title || detail;
      if (Array.isArray(problem.errors) && problem.errors.length) {
        detail = problem.errors.map((e) => `${e.field}: ${e.message}`).join("; ");
      }
    } catch {
      /* a proxy error page is not JSON; keep the status-based message */
    }
    if (response.status === 401) {
      detail =
        "The provenance service rejected this session. It may not be configured " +
        "to trust iMATCH tokens (IIE_IMATCH_JWT_SECRET must match NEXGEN_JWT_SECRET).";
    }
    throw new Error(detail);
  }

  if (response.status === 204) return null;
  return response.json();
}

// -- investigations ---------------------------------------------------------

export function listInvestigations() {
  return request("/investigations");
}

export function createInvestigation({ caseId, title, lawfulBasis, purpose = "" }) {
  return request("/investigations", {
    method: "POST",
    body: {
      case_id: caseId,
      title,
      lawful_basis: lawfulBasis,
      purpose,
      jurisdiction: "IN",
    },
  });
}

export function uploadImage(investigationId, file) {
  const form = new FormData();
  form.append("file", file);
  return request(`/investigations/${investigationId}/images`, {
    method: "POST",
    body: form,
  });
}

export function startRun(investigationId) {
  return request(`/investigations/${investigationId}/runs`, {
    method: "POST",
    body: { trigger: "MANUAL" },
  });
}

export function getRun(runId) {
  return request(`/runs/${runId}`);
}

// -- discovery and findings -------------------------------------------------

/** Every provider and whether it can actually run — reported *before* a search,
 *  so an investigator knows what coverage to expect rather than inferring it
 *  from an empty result. */
export function listProviders() {
  return request("/providers");
}

/**
 * Ask the configured providers where this image appears.
 *
 * `urls` are operator-supplied leads for the `manual` provider: pages the
 * investigator already suspects. Targeted corroboration, never a crawl seed.
 */
export function discover(investigationId, { urls = [] } = {}) {
  return request(`/investigations/${investigationId}/discover`, {
    method: "POST",
    body: { urls },
  });
}

/** What has been found so far, without running anything new. */
export function getFindings(investigationId) {
  return request(`/investigations/${investigationId}/findings`);
}

/**
 * Open a provenance investigation for one image, end to end.
 *
 * Four calls rather than one, because each is separately meaningful in the
 * audit chain: opening a case records the lawful basis, the upload opens the
 * image's chain of custody, and starting a run records which stages will run
 * under which configuration.
 *
 * `onStep` reports progress so the UI can show what is happening instead of
 * freezing on a single spinner for the whole sequence.
 */
export async function traceImage({ file, caseRef, title, lawfulBasis, purpose, onStep }) {
  onStep?.("Opening provenance case…");
  const investigation = await createInvestigation({
    caseId: caseRef,
    title,
    lawfulBasis,
    purpose,
  });

  onStep?.("Uploading image and hashing…");
  const upload = await uploadImage(investigation.id, file);

  onStep?.("Starting discovery run…");
  const run = await startRun(investigation.id);

  return { investigation, image: upload.image, deduplicated: upload.deduplicated, run };
}

/**
 * Live stage progress over Server-Sent Events.
 *
 * `EventSource` cannot set an Authorization header, so the stream is read with
 * fetch + a ReadableStream reader. IIE sends a complete snapshot on connect
 * rather than only deltas, so joining late — or reconnecting after a sleep —
 * shows true state immediately instead of an empty bar for a run already
 * underway.
 *
 * Returns an abort function; call it on unmount.
 */
export function streamRunProgress(runId, { onProgress, onEnd, onError } = {}) {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API}/runs/${runId}/events`, {
        headers: { Authorization: `Bearer ${tokenStore.access || ""}` },
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(`Progress stream unavailable (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // Frames are separated by a blank line. Whatever follows the final
        // separator is a partial frame and stays buffered.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";

        for (const frame of frames) {
          const lines = frame.split("\n");
          const dataLine = lines.find((line) => line.startsWith("data: "));
          if (!dataLine) continue; // keepalive comment frame
          const isEnd = lines.some((line) => line.startsWith("event: end"));
          const payload = JSON.parse(dataLine.slice(6));
          if (isEnd) onEnd?.(payload.status);
          else onProgress?.(payload);
        }
      }
    } catch (error) {
      if (error.name !== "AbortError") onError?.(error);
    }
  })();

  return () => controller.abort();
}
