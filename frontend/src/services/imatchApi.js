const LOCAL_BIOMETRICS_BASE = "http://127.0.0.1:8000/api/biometrics";

export const identifyApiUrl =
  import.meta.env.VITE_IDENTIFY_API_URL?.trim() || `${LOCAL_BIOMETRICS_BASE}/identify`;

export const verifyApiUrl =
  import.meta.env.VITE_VERIFY_API_URL?.trim() || `${LOCAL_BIOMETRICS_BASE}/verify`;

export const batchIdentifyApiUrl =
  import.meta.env.VITE_BATCH_API_URL?.trim() || `${LOCAL_BIOMETRICS_BASE}/batch-identify`;

export const enrollApiUrl =
  import.meta.env.VITE_ENROLL_API_URL?.trim() || `${LOCAL_BIOMETRICS_BASE}/enroll`;

/**
 * 1:N Single Face Search — calls real /api/biometrics/identify endpoint.
 */
export async function runImatchSearch({ file, mode, sourceUrl, checks }) {
  if (!file && !sourceUrl?.trim()) {
    throw new Error("Please select or upload a face image file first.");
  }

  const form = new FormData();
  if (file) {
    // Basic file type validation
    if (!file.type.startsWith("image/")) {
      throw new Error("Invalid file type: Selected file is not an image.");
    }
    form.append("file", file, file.name);
  }
  form.append("top_k", "5");
  form.append("operator_id", "demo_operator");

  let response;
  try {
    response = await fetch(identifyApiUrl, {
      method: "POST",
      body: form,
    });
  } catch (netErr) {
    throw new Error(`Backend server unavailable at ${identifyApiUrl}. Please verify the FastAPI backend server is running on port 8000.`);
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || "Biometric identification request failed.");
  }

  return normalizeImatchResult(payload);
}

/**
 * 1:1 Face Comparison — calls real /api/biometrics/verify endpoint.
 */
export async function runVerifyCompare(referenceFile, probeFile) {
  if (!referenceFile) throw new Error("Upload a reference (first) face image.");
  if (!probeFile) throw new Error("Upload a probe (second) face image.");

  if (!referenceFile.type.startsWith("image/")) {
    throw new Error("Reference file is not a valid image format.");
  }
  if (!probeFile.type.startsWith("image/")) {
    throw new Error("Probe file is not a valid image format.");
  }

  const form = new FormData();
  form.append("reference", referenceFile, referenceFile.name);
  form.append("probe", probeFile, probeFile.name);
  form.append("operator_id", "demo_operator");

  let response;
  try {
    response = await fetch(verifyApiUrl, {
      method: "POST",
      body: form,
    });
  } catch (netErr) {
    throw new Error(`Backend server unavailable at ${verifyApiUrl}. Please verify the FastAPI backend server is running on port 8000.`);
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || "Face comparison request failed.");
  }

  return {
    score: Number(payload.score ?? 0),
    label: payload.label ?? "unknown",
    verified: Boolean(payload.verified),
    reviewRequired: Boolean(payload.review_required),
    qualityRef: Number(payload.quality_ref ?? 0),
    qualityProbe: Number(payload.quality_probe ?? 0),
    livenessRef: Number(payload.liveness_ref ?? 0),
    livenessProbe: Number(payload.liveness_probe ?? 0),
    reasonsRef: payload.reasons_ref ?? [],
    reasonsProbe: payload.reasons_probe ?? [],
    auditHash: payload.audit_hash ?? "",
    thresholds: payload.thresholds ?? { same_person: 0.42, inconclusive_low: 0.28 },
  };
}

/**
 * Batch 1:N Face Search — calls real /api/biometrics/batch-identify endpoint.
 */
export async function runBatchIdentify(files) {
  if (!files || files.length === 0) {
    throw new Error("Select at least one face image for batch processing.");
  }

  const form = new FormData();
  for (const f of files) {
    if (!f.type.startsWith("image/")) {
      throw new Error(`Invalid file '${f.name}': Not a valid image file.`);
    }
    form.append("files", f, f.name);
  }
  form.append("top_k", "5");
  form.append("operator_id", "demo_operator");

  let response;
  try {
    response = await fetch(batchIdentifyApiUrl, {
      method: "POST",
      body: form,
    });
  } catch (netErr) {
    throw new Error(`Backend server unavailable at ${batchIdentifyApiUrl}. Please verify the FastAPI backend server is running on port 8000.`);
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || "Batch face search request failed.");
  }

  return payload;
}

function normalizeImatchResult(payload) {
  const quality = Number(payload?.quality_score ?? payload?.quality?.score ?? 0);
  const liveness = Number(payload?.liveness_score ?? payload?.liveness?.score ?? 0);
  const matches = payload?.matches || [];
  const topMatchScore = matches.length > 0 ? Number(matches[0].confidence ?? 0) : 0;

  return {
    decision: payload?.decision || "analysis_complete",
    quality: clampScore(quality),
    liveness: clampScore(liveness),
    matchScore: clampScore(topMatchScore),
    reviewRequired: Boolean(payload?.review_required),
    matches: matches.slice(0, 5).map((match, index) => ({
      id: match.identity_id || `Candidate ${index + 1}`,
      score: clampScore(Number(match.confidence ?? match.score ?? 0)),
      metadata: match.metadata || {},
    })),
    auditHash: payload?.audit_hash || "",
  };
}

function clampScore(value) {
  if (!Number.isFinite(value)) return 0;
  if (value > 1) return Math.max(0, Math.min(1, value / 100));
  return Math.max(0, Math.min(1, value));
}
