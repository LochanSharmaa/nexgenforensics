/**
 * HTTP client for the iMATCH API.
 *
 * In development VITE_IMATCH_API_BASE is left empty and requests go to a
 * relative /api path, which the Vite dev proxy forwards to the backend. That
 * keeps the browser same-origin, so there is no CORS grant and no cross-origin
 * biometric request.
 */

const configuredBase = (import.meta.env.VITE_IMATCH_API_BASE || "").trim().replace(/\/$/, "");

export const apiBase = configuredBase;

const ACCESS_TOKEN_KEY = "imatch.access_token";
const REFRESH_TOKEN_KEY = "imatch.refresh_token";

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }

  get isAuthError() {
    return this.status === 401;
  }

  get isRateLimited() {
    return this.status === 429;
  }
}

/**
 * Tokens live in sessionStorage, not localStorage: an investigator's session
 * should not outlive the browser tab, and this limits the window in which a
 * shared or unattended workstation exposes a valid credential.
 */
export const tokenStore = {
  get access() {
    return sessionStorage.getItem(ACCESS_TOKEN_KEY);
  },
  get refresh() {
    return sessionStorage.getItem(REFRESH_TOKEN_KEY);
  },
  set(accessToken, refreshToken) {
    sessionStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
    if (refreshToken) sessionStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
  },
  clear() {
    sessionStorage.removeItem(ACCESS_TOKEN_KEY);
    sessionStorage.removeItem(REFRESH_TOKEN_KEY);
  },
};

function assertSecureEndpoint() {
  if (import.meta.env.PROD && apiBase && !apiBase.startsWith("https://")) {
    throw new ApiError("The production API endpoint must use HTTPS.", 0, null);
  }
}

function describeError(status, payload) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  if (Array.isArray(detail) && detail.length) {
    // FastAPI validation errors arrive as a list of {loc, msg}.
    return detail.map((item) => item.msg || String(item)).join("; ");
  }
  if (status === 401) return "Your session has expired. Sign in again.";
  if (status === 403) return "You do not have permission to do that.";
  if (status === 404) return "Not found.";
  if (status === 429) return "Too many requests. Wait a moment and try again.";
  if (status >= 500) return "The service reported an internal error.";
  return `Request failed with status ${status}.`;
}

let refreshInFlight = null;

async function refreshAccessToken() {
  const refreshToken = tokenStore.refresh;
  if (!refreshToken) return false;

  // Collapse concurrent refreshes: several parallel requests hitting a just
  // expired token would otherwise each burn a refresh round-trip.
  if (!refreshInFlight) {
    refreshInFlight = (async () => {
      try {
        const response = await fetch(`${apiBase}/api/auth/refresh`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!response.ok) return false;
        const body = await response.json();
        tokenStore.set(body.access_token, body.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        refreshInFlight = null;
      }
    })();
  }

  return refreshInFlight;
}

async function parseBody(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return null;
    }
  }
  return await response.text();
}

/**
 * Perform an authenticated request, retrying once after a token refresh.
 */
export async function request(path, { method = "GET", body, headers = {}, auth = true, retry = true } = {}) {
  assertSecureEndpoint();

  const finalHeaders = { ...headers };
  if (body !== undefined) finalHeaders["Content-Type"] = "application/json";
  if (auth && tokenStore.access) finalHeaders.Authorization = `Bearer ${tokenStore.access}`;

  let response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      method,
      headers: finalHeaders,
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "Could not reach the iMATCH service. Check that the backend is running on port 8443.",
      0,
      null,
    );
  }

  if (response.status === 401 && auth && retry && (await refreshAccessToken())) {
    return request(path, { method, body, headers, auth, retry: false });
  }

  const payload = await parseBody(response);

  if (!response.ok) {
    throw new ApiError(describeError(response.status, payload), response.status, payload);
  }

  return payload;
}

/** Read a File into a bare base64 string, without the data: URL prefix. */
export function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}
