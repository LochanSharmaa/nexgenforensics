/**
 * NexGen iMATCH API client.
 *
 * THE ONLY BACKEND IS `imatch_api` ON :8443 — what deployment/Dockerfile runs.
 * An earlier version of this file spoke a different contract entirely:
 * multipart uploads to /api/biometrics/* on :8000. That belonged to
 * backend/app/, a package from an unrelated product that could not even start
 * and is now quarantined in backend/_deprecated_app/. Every call here targets
 * a route confirmed present in the live /openapi.json.
 *
 * Contract, in one place so it is not rediscovered per-function:
 *   - Bearer token auth (Authorization: Bearer <access_token>)
 *   - JSON bodies; images are base64 WITHOUT a data: URL prefix
 *   - `lawful_basis` is required on search and verify. The server cannot judge
 *     whether a search was lawful, but it records verbatim that someone had to
 *     state a reason. Do not default it to a placeholder — make the operator
 *     type it.
 *   - Errors surface the server's real message, including FastAPI's 422
 *     validation arrays, rather than a generic "request failed".
 */

/**
 * Development API origin.
 *
 * Derived from the page's own hostname rather than hardcoding 127.0.0.1,
 * because `localhost` and `127.0.0.1` are DIFFERENT SITES to a browser even
 * though they are the same machine. With the page on localhost:5173 and the
 * API pinned to 127.0.0.1:8443, a SameSite=Lax cookie is not sent on POST, so
 * the CSRF double-submit check has nothing to compare and every form fails.
 * Ports are irrelevant to SameSite, so matching the hostname is enough to make
 * the two same-site.
 *
 * DEVELOPMENT ONLY. This used to be the fallback in every environment, and on
 * the deployed site it resolved to https://<the-vercel-domain>:8443 — a port
 * nothing listens on, at an origin the Content-Security-Policy does not allow.
 * That was the "Failed to fetch" every login returned.
 */
const DEV_BASE =
  typeof window !== "undefined" && window.location?.hostname
    ? `${window.location.protocol}//${window.location.hostname}:8443`
    : "http://127.0.0.1:8443";

/**
 * VITE_IMATCH_API_BASE is the documented name — it is what frontend/.env.example
 * describes and what frontend/.env.production sets. This module previously read
 * only VITE_IMATCH_BASE_URL, a name nothing set and no example mentioned, so
 * configuring the documented variable had no effect on the client every page
 * actually imports. Both are accepted now; the documented one wins.
 */
const rawBase = (
  import.meta.env.VITE_IMATCH_API_BASE ||
  import.meta.env.VITE_IMATCH_BASE_URL ||
  ""
).trim();

/**
 * "same-origin" (or "/") is the self-hosted deployment: the same server that
 * serves this bundle proxies /api to the backend (deployment/nginx). Requests
 * then use relative paths — same origin, so no CORS grant is needed and the
 * CSRF cookie is first-party. This is the recommended mode on a personal
 * server; an absolute https:// origin is only for split hosting (e.g. the
 * frontend on a CDN with the API elsewhere).
 */
const sameOrigin = rawBase === "/" || rawBase.toLowerCase() === "same-origin";

const configuredBase = sameOrigin ? "" : rawBase.replace(/\/$/, "");

// Guessing an origin in production is what produced a bundle that pointed at a
// port on the static host. Refuse to guess: fail at load, where the message
// names the cause, rather than at the first login with "Failed to fetch".
if (import.meta.env.PROD && !rawBase) {
  throw new Error(
    "VITE_IMATCH_API_BASE is not set in this production build, so the bundle " +
      "has no API origin. Set it to 'same-origin' for a single-server deploy, " +
      "or to the API's https:// origin for split hosting, and rebuild.",
  );
}

export const IMATCH_BASE = sameOrigin ? "" : configuredBase || DEV_BASE;

/** Shown in the UI so an operator can see which backend they are hitting. */
export const imatchApiUrl = `${IMATCH_BASE}/api/imatch`;

// ---------------------------------------------------------------- tokens ---

const ACCESS_KEY = "nexgen.imatch.access";
const REFRESH_KEY = "nexgen.imatch.refresh";

/**
 * Token storage.
 *
 * sessionStorage, not localStorage: biometric tooling should not leave a
 * usable session token on disk after the browser closes. This is still
 * XSS-reachable — the durable fix is httpOnly cookies, which needs a server
 * change and is not done here.
 */
export const tokenStore = {
  get access() {
    try {
      return sessionStorage.getItem(ACCESS_KEY);
    } catch {
      return null;
    }
  },
  get refresh() {
    try {
      return sessionStorage.getItem(REFRESH_KEY);
    } catch {
      return null;
    }
  },
  set(accessToken, refreshToken) {
    try {
      if (accessToken) sessionStorage.setItem(ACCESS_KEY, accessToken);
      if (refreshToken) sessionStorage.setItem(REFRESH_KEY, refreshToken);
    } catch {
      /* storage disabled; session simply will not persist across reloads */
    }
  },
  clear() {
    try {
      sessionStorage.removeItem(ACCESS_KEY);
      sessionStorage.removeItem(REFRESH_KEY);
    } catch {
      /* nothing to clear */
    }
  },
};

// ----------------------------------------------------------------- error ---

export class ApiError extends Error {
  constructor(message, { status = 0, detail = null, url = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.url = url;
  }
}

/**
 * Turn a FastAPI error body into something an operator can act on.
 *
 * FastAPI returns 422 as `detail: [{loc, msg, type}, ...]`. Rendering that
 * object stringifies to "[object Object]", which tells the user nothing, so
 * the field path and message are extracted explicitly.
 */
function describeError(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === "string" && detail.trim()) return detail;

  if (Array.isArray(detail)) {
    const parts = detail
      .map((d) => {
        const field = Array.isArray(d?.loc)
          ? d.loc.filter((x) => x !== "body").join(".")
          : "";
        const msg = d?.msg || d?.message || "invalid value";
        return field ? `${field}: ${msg}` : msg;
      })
      .filter(Boolean);
    if (parts.length) return parts.join("; ");
  }

  if (typeof payload?.message === "string" && payload.message) return payload.message;

  if (status === 401) return "Not signed in, or the session has expired.";
  if (status === 403) return "Your role does not permit this action.";
  if (status === 404) return "Not found.";
  if (status === 429) return "Rate limit reached. Wait a moment and retry.";
  if (status >= 500) return `Server error (${status}).`;
  return `Request failed (${status}).`;
}

// --------------------------------------------------------------- request ---

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
let csrfToken = null;
let csrfInFlight = null;

/**
 * Fetch a CSRF token, once, and reuse it.
 *
 * Only needed for state-changing requests that carry no Authorization header.
 * A request with a bearer token is exempt server-side, because a cross-origin
 * page cannot set that header in the first place — so fetching a token for
 * those would be a wasted round trip on every call.
 *
 * `credentials: "include"` matters: the API is on a different origin to the
 * app in development, and without it the browser sends no cookie, so the
 * server's double-submit check has nothing to compare the header against.
 *
 * Concurrent callers share one request. Letting two run means two tokens are
 * minted and the second overwrites the first's cookie, so whichever request
 * carries the older header fails the double-submit check -- a 403 on a form the
 * user filled in correctly, and one that only shows up when two actions start
 * together.
 */
async function ensureCsrfToken() {
  if (csrfToken) return csrfToken;
  if (csrfInFlight) return csrfInFlight;

  csrfInFlight = (async () => {
    const response = await fetch(`${IMATCH_BASE}/api/auth/csrf`, { credentials: "include" });
    if (!response.ok) return null;
    const payload = await response.json();
    csrfToken = payload?.csrf_token ?? null;
    return csrfToken;
  })().finally(() => {
    csrfInFlight = null;
  });

  return csrfInFlight;
}

async function request(path, { method = "GET", body, auth = true, signal } = {}) {
  const url = `${IMATCH_BASE}${path}`;
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  const bearer = auth && tokenStore.access;
  if (bearer) {
    headers.Authorization = `Bearer ${tokenStore.access}`;
  }

  if (!SAFE_METHODS.has(method.toUpperCase()) && !bearer) {
    const token = await ensureCsrfToken();
    if (token) headers["X-CSRF-Token"] = token;
  }

  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
      credentials: "include",
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    });
  } catch (networkError) {
    if (networkError?.name === "AbortError") throw networkError;
    throw new ApiError(
      `Cannot reach the iMATCH API at ${IMATCH_BASE}. Confirm the backend is ` +
        `running:  python -m uvicorn imatch_api.main:app --host 127.0.0.1 --port 8443 --app-dir backend`,
      { url },
    );
  }

  if (response.status === 204) return null;

  let payload = null;
  const text = await response.text();
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text.slice(0, 500) };
    }
  }

  if (!response.ok) {
    // 401 means the stored token is dead. Clear it so the app stops presenting
    // the operator as signed in while every request fails.
    if (response.status === 401) tokenStore.clear();
    // A stale CSRF token (expired, or minted against a restarted server) would
    // otherwise wedge every form on the page permanently. Drop it so the next
    // attempt fetches a fresh one.
    if (response.status === 403 && /csrf/i.test(payload?.detail ?? "")) {
      csrfToken = null;
    }
    throw new ApiError(describeError(payload, response.status), {
      status: response.status,
      detail: payload?.detail ?? null,
      url,
    });
  }

  return payload;
}

// ---------------------------------------------------------------- images ---

/**
 * Read a File into base64 with NO `data:` prefix.
 *
 * The API takes raw base64. Sending the data-URL form makes the server's
 * decoder fail on a header it did not ask for, so the prefix is stripped here
 * rather than in each caller.
 */
export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new ApiError("No file supplied."));
      return;
    }
    if (!file.type?.startsWith("image/")) {
      reject(new ApiError(`"${file.name}" is not an image file.`));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new ApiError(`Could not read "${file.name}".`));
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.readAsDataURL(file);
  });
}

// ------------------------------------------------------------------ auth ---

export async function login({ email, password, tenant = "", rememberMe = false }) {
  const payload = await request("/api/auth/login", {
    method: "POST",
    auth: false,
    body: { email, password, tenant, remember_me: rememberMe },
  });
  tokenStore.set(payload?.access_token, payload?.refresh_token);
  // The login response carries tokens, not a profile. Fetch the profile so the
  // caller always receives the same shape it would get on session restore.
  return fetchCurrentUser();
}

/**
 * Account lifecycle. Every one of these is unauthenticated (auth: false) --
 * they are the endpoints a person uses precisely because they cannot sign in
 * yet, so attaching a bearer token would be meaningless.
 */
export async function register({ fullName, email, password, confirmPassword, tenant = "" }) {
  return request("/api/auth/register", {
    method: "POST",
    auth: false,
    body: {
      full_name: fullName,
      email,
      password,
      confirm_password: confirmPassword,
      tenant,
    },
  });
}

export async function verifyEmail({ email, otp }) {
  return request("/api/auth/verify-email", {
    method: "POST",
    auth: false,
    body: { email, otp },
  });
}

export async function resendOtp({ email }) {
  return request("/api/auth/resend-otp", { method: "POST", auth: false, body: { email } });
}

export async function forgotPassword({ email }) {
  return request("/api/auth/forgot-password", { method: "POST", auth: false, body: { email } });
}

export async function resetPassword({ token, password, confirmPassword }) {
  return request("/api/auth/reset-password", {
    method: "POST",
    auth: false,
    body: { token, password, confirm_password: confirmPassword },
  });
}

export async function logout() {
  try {
    await request("/api/auth/logout", { method: "POST" });
  } catch {
    // A failed server-side logout must not strand the operator in a
    // half-signed-in UI. The local token is cleared regardless.
  } finally {
    tokenStore.clear();
  }
}

export function fetchCurrentUser() {
  return request("/api/auth/me");
}

// ----------------------------------------------------------------- cases ---

export function listCases() {
  return request("/api/cases");
}

export function createCase(payload) {
  return request("/api/cases", { method: "POST", body: payload });
}

export function getCase(caseId) {
  return request(`/api/cases/${encodeURIComponent(caseId)}`);
}

export function updateCase(caseId, payload) {
  return request(`/api/cases/${encodeURIComponent(caseId)}`, {
    method: "PATCH",
    body: payload,
  });
}

/**
 * Fetch a case report in the caller's chosen format.
 *
 * JSON goes through `request()` as usual. Markdown and PDF are NOT JSON — the
 * generic helper would mangle both (parse-fail a .md body into `{detail}`,
 * corrupt PDF bytes via text decoding) — so they use a direct fetch and return
 * text or a Blob respectively, ready to hand to a download link.
 */
export async function fetchCaseReport(caseId, fmt = "json") {
  const path = `/api/cases/${encodeURIComponent(caseId)}/report`;
  if (fmt === "json") return request(path);

  const response = await fetch(`${IMATCH_BASE}${path}?fmt=${encodeURIComponent(fmt)}`, {
    headers: tokenStore.access ? { Authorization: `Bearer ${tokenStore.access}` } : {},
    credentials: "include",
  });
  if (!response.ok) {
    if (response.status === 401) tokenStore.clear();
    let detail = null;
    try {
      detail = (await response.json())?.detail;
    } catch {
      /* non-JSON error body; the status is the message */
    }
    throw new ApiError(detail || `Report export failed (HTTP ${response.status}).`, {
      status: response.status,
    });
  }
  return fmt === "pdf" ? response.blob() : response.text();
}

// -------------------------------------------------------------- subjects ---

export async function enrolSubject({
  file,
  imageBase64,
  displayName = "",
  externalRef = "",
  notes = "",
  caseId = null,
  subjectId = null,
}) {
  const image_base64 = imageBase64 ?? (await fileToBase64(file));
  return request("/api/subjects", {
    method: "POST",
    body: {
      image_base64,
      display_name: displayName,
      external_ref: externalRef,
      notes,
      case_id: caseId,
      subject_id: subjectId,
    },
  });
}

export function listSubjects() {
  return request("/api/subjects");
}

export function deleteSubject(subjectId) {
  return request(`/api/subjects/${encodeURIComponent(subjectId)}`, { method: "DELETE" });
}

export function listSubjectTemplates(subjectId) {
  return request(`/api/subjects/${encodeURIComponent(subjectId)}/templates`);
}

// ------------------------------------------------------- search / verify ---

/** 1:N search against the caller's tenant gallery. */
export async function runSearch({
  file,
  imageBase64,
  lawfulBasis,
  caseId = null,
  topK = 10,
  mode = "single",
}) {
  const image_base64 = imageBase64 ?? (await fileToBase64(file));
  return request("/api/imatch/search", {
    method: "POST",
    body: {
      image_base64,
      lawful_basis: lawfulBasis ?? "",
      case_id: caseId,
      top_k: topK,
      mode,
    },
  });
}

/** 1:1 verification of two supplied images. Nothing is enrolled. */
export async function runVerification({
  referenceFile,
  probeFile,
  referenceBase64,
  probeBase64,
  lawfulBasis,
  caseId = null,
}) {
  const reference_image_base64 = referenceBase64 ?? (await fileToBase64(referenceFile));
  const probe_image_base64 = probeBase64 ?? (await fileToBase64(probeFile));
  return request("/api/imatch/verify", {
    method: "POST",
    body: {
      reference_image_base64,
      probe_image_base64,
      lawful_basis: lawfulBasis ?? "",
      case_id: caseId,
    },
  });
}

/**
 * Batch comparison. Three modes, matching POST /api/imatch/batch:
 *
 *   one_to_many  ONE reference vs every probe. The common investigative case
 *                ("here is my suspect, check these 30 stills"). The reference
 *                is sent once and encoded once, not per item.
 *   pair         each item carries its OWN reference; N independent 1:1
 *                comparisons of different couples.
 *   gallery      each probe searched against the enrolled gallery.
 *
 * One unreadable file does not fail the batch: the server isolates per item
 * and returns `status: "error"` for that entry only.
 */
export async function runBatch({
  mode = "one_to_many",
  referenceFile = null,
  probeFiles = [],
  lawfulBasis,
  caseId = null,
  topK = 5,
}) {
  if (!probeFiles.length) {
    throw new ApiError("Select at least one image to compare.");
  }
  if (probeFiles.length > 50) {
    throw new ApiError(
      `Batch limit is 50 images; ${probeFiles.length} selected. ` +
        "Split it into smaller batches.",
    );
  }
  if (mode === "one_to_many" && !referenceFile) {
    throw new ApiError("Select a reference image to compare every upload against.");
  }

  const body = {
    mode,
    lawful_basis: lawfulBasis ?? "",
    case_id: caseId,
    top_k: topK,
    items: await Promise.all(
      probeFiles.map(async (f) => ({
        label: f.name,
        probe_image_base64: await fileToBase64(f),
      })),
    ),
  };
  if (mode === "one_to_many") {
    body.reference_image_base64 = await fileToBase64(referenceFile);
  }
  return request("/api/imatch/batch", { method: "POST", body });
}

export function listSearches(caseId) {
  const q = caseId ? `?case_id=${encodeURIComponent(caseId)}` : "";
  return request(`/api/imatch/searches${q}`);
}

export function listCandidates(searchId) {
  return request(`/api/imatch/searches/${encodeURIComponent(searchId)}/candidates`);
}

export function adjudicateCandidate(candidateId, adjudication, examinerNotes = "") {
  return request(`/api/imatch/candidates/${encodeURIComponent(candidateId)}/adjudicate`, {
    method: "POST",
    body: { adjudication, examiner_notes: examinerNotes },
  });
}

// ------------------------------------------------------------ audit/engine ---

export function listAuditRecords(params = {}) {
  const q = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ""),
  ).toString();
  return request(`/api/audit${q ? `?${q}` : ""}`);
}

export function verifyAuditChain() {
  return request("/api/audit/verify");
}

export function fetchEngineStatus() {
  return request("/api/imatch/engine/status");
}

export function fetchHealth() {
  return request("/api/health", { auth: false });
}

// ------------------------------------------------------------- normalize ---

/**
 * Flatten a /api/imatch/verify response for display.
 *
 * `liveness` and `deepfakeRisk` are carried through with their heuristic
 * qualifiers intact. The backend reports `certified: false` and
 * `method: passive_single_frame_heuristic` on every liveness block; any UI
 * rendering these MUST keep that visible. They are image-quality signals, not
 * presentation-attack detection, and must never be labelled as spoof checks.
 */
export function normalizeVerifyResult(payload) {
  const side = (s) => ({
    quality: Number(s?.quality?.score ?? 0),
    qualityAccepted: Boolean(s?.quality?.accepted),
    qualityReasons: s?.quality?.reasons ?? [],
    liveness: Number(s?.liveness?.score ?? 0),
    livenessPassed: Boolean(s?.liveness?.passed),
    livenessCertified: Boolean(s?.liveness?.certified), // always false today
    livenessMethod: s?.liveness?.method ?? "unknown",
    livenessReasons: s?.liveness?.reasons ?? [],
    deepfakeRisk: Number(s?.deepfake_risk ?? 0),
    facesDetected: Number(s?.faces_detected ?? 0),
    detector: s?.detector ?? "",
    pose: s?.pose ?? null,
  });

  return {
    similarity: Number(payload?.similarity ?? 0),
    verified: Boolean(payload?.verified),
    threshold: typeof payload?.threshold === "number" ? payload.threshold : null,
    explanation: payload?.explanation ?? "",
    recognitionCapable: Boolean(payload?.recognition_capable),
    reference: side(payload?.reference),
    probe: side(payload?.probe),
    morphing: payload?.morphing ?? null,
    auditHash: payload?.audit_hash ?? "",
    notice: payload?.notice ?? "",
  };
}
