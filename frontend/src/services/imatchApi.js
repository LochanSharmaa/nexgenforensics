import { fileToBase64, request, tokenStore } from "./apiClient";

export { ApiError, apiBase, fileToBase64, tokenStore } from "./apiClient";

// ------------------------------------------------------------------ auth ----

export async function login({ email, password, tenant }) {
  const body = await request("/api/auth/login", {
    method: "POST",
    auth: false,
    body: { email, password, tenant: tenant || "" },
  });
  tokenStore.set(body.access_token, body.refresh_token);
  return body.user;
}

export async function logout() {
  try {
    await request("/api/auth/logout", { method: "POST" });
  } finally {
    // Clear locally even if the server call fails, so a network problem can
    // never leave a credential sitting in the tab.
    tokenStore.clear();
  }
}

export function fetchCurrentUser() {
  return request("/api/auth/me");
}

// ----------------------------------------------------------------- cases ----

export function listCases(status) {
  const query = status ? `?status_filter=${encodeURIComponent(status)}` : "";
  return request(`/api/cases${query}`);
}

export function getCase(caseId) {
  return request(`/api/cases/${caseId}`);
}

export function createCase({ reference, title, description, lawfulBasis }) {
  return request("/api/cases", {
    method: "POST",
    body: {
      reference,
      title,
      description: description || "",
      lawful_basis: lawfulBasis || "",
    },
  });
}

export function updateCase(caseId, changes) {
  return request(`/api/cases/${caseId}`, { method: "PATCH", body: changes });
}

export function caseReportUrl(caseId, format = "json") {
  return `/api/cases/${caseId}/report?fmt=${format}`;
}

export function fetchCaseReport(caseId, format = "json") {
  return request(caseReportUrl(caseId, format));
}

// -------------------------------------------------------------- subjects ----

export function listSubjects(caseId) {
  const query = caseId ? `?case_id=${encodeURIComponent(caseId)}` : "";
  return request(`/api/subjects${query}`);
}

export async function enrolSubject({ file, displayName, externalRef, notes, caseId, subjectId, lawfulBasis }) {
  if (!file) throw new Error("Select an enrolment image first.");
  return request("/api/subjects", {
    method: "POST",
    body: {
      display_name: displayName || "",
      external_ref: externalRef || "",
      notes: notes || "",
      case_id: caseId || null,
      subject_id: subjectId || null,
      image_base64: await fileToBase64(file),
      lawful_basis: lawfulBasis || "",
    },
  });
}

export function deleteSubject(subjectId) {
  return request(`/api/subjects/${subjectId}`, { method: "DELETE" });
}

export function listSubjectTemplates(subjectId) {
  return request(`/api/subjects/${subjectId}/templates`);
}

// ---------------------------------------------------------------- search ----

export async function runSearch({ file, caseId, lawfulBasis, purpose, topK = 10, mode = "single", checks = [] }) {
  if (!file) throw new Error("Select a probe image first.");
  return request("/api/imatch/search", {
    method: "POST",
    body: {
      image_base64: await fileToBase64(file),
      mode,
      case_id: caseId || null,
      top_k: topK,
      lawful_basis: lawfulBasis || "",
      purpose: purpose || "",
      checks,
    },
  });
}

export async function runVerification({ referenceFile, probeFile, caseId, lawfulBasis }) {
  if (!referenceFile || !probeFile) {
    throw new Error("Both a reference and a probe image are required.");
  }
  const [reference, probe] = await Promise.all([fileToBase64(referenceFile), fileToBase64(probeFile)]);
  return request("/api/imatch/verify", {
    method: "POST",
    body: {
      reference_image_base64: reference,
      probe_image_base64: probe,
      case_id: caseId || null,
      lawful_basis: lawfulBasis || "",
    },
  });
}

export function listSearches(caseId, limit = 50) {
  const params = new URLSearchParams();
  if (caseId) params.set("case_id", caseId);
  params.set("limit", String(limit));
  return request(`/api/imatch/searches?${params}`);
}

export function listCandidates(searchId) {
  return request(`/api/imatch/searches/${searchId}/candidates`);
}

export function adjudicateCandidate(candidateId, { adjudication, notes }) {
  return request(`/api/imatch/candidates/${candidateId}/adjudicate`, {
    method: "POST",
    body: { adjudication, examiner_notes: notes || "" },
  });
}

// ----------------------------------------------------------------- audit ----

export function listAuditRecords({ action, resourceId, limit = 100 } = {}) {
  const params = new URLSearchParams();
  if (action) params.set("action", action);
  if (resourceId) params.set("resource_id", resourceId);
  params.set("limit", String(limit));
  return request(`/api/audit?${params}`);
}

export function verifyAuditChain() {
  return request("/api/audit/verify");
}

// ---------------------------------------------------------------- system ----

export function fetchEngineStatus() {
  return request("/api/imatch/engine/status");
}

export function fetchHealth() {
  return request("/api/health", { auth: false });
}
