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

const DEFAULT_BASE = "http://127.0.0.1:8443";

export const IMATCH_BASE =
  import.meta.env.VITE_IMATCH_BASE_URL?.trim() || DEFAULT_BASE;

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

async function request(path, { method = "GET", body, auth = true, signal } = {}) {
  const url = `${IMATCH_BASE}${path}`;
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (auth && tokenStore.access) {
    headers.Authorization = `Bearer ${tokenStore.access}`;
  }

  let response;
  try {
    response = await fetch(url, {
      method,
      headers,
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

export function fetchCaseReport(caseId) {
  return request(`/api/cases/${encodeURIComponent(caseId)}/report`);
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
