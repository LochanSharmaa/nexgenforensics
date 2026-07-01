const DEFAULT_IMATCH_API_URL = "http://localhost:8443/api/imatch/search";

export const imatchApiUrl =
  import.meta.env.VITE_IMATCH_API_URL?.trim() || DEFAULT_IMATCH_API_URL;

export async function runImatchSearch({ file, mode, sourceUrl, checks }) {
  if (!file && !sourceUrl?.trim()) {
    throw new Error("Upload a photo or enter a secure image URL first.");
  }

  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("purpose", "authorized_imatch_demo");
  formData.append("lawful_use_reason", "User-submitted iMatch page analysis");
  formData.append("checks", JSON.stringify(checks));

  if (file) {
    formData.append("image", file, file.name);
  }

  if (sourceUrl?.trim()) {
    formData.append("source_url", sourceUrl.trim());
  }

  const response = await fetch(imatchApiUrl, {
    method: "POST",
    body: formData,
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || "iMatch AI service rejected the request.");
  }

  return normalizeImatchResult(payload);
}

function normalizeImatchResult(payload) {
  const quality = Number(payload?.quality?.score ?? payload?.quality_score ?? 0);
  const liveness = Number(payload?.liveness?.score ?? payload?.liveness_score ?? 0);
  const matchScore = Number(payload?.match?.score ?? payload?.score ?? 0);
  const matches = payload?.matches || payload?.candidates || [];

  return {
    decision: payload?.decision || payload?.match?.decision || "analysis_complete",
    quality: clampScore(quality),
    liveness: clampScore(liveness),
    matchScore: clampScore(matchScore),
    reviewRequired: Boolean(payload?.review_required || payload?.human_review_required),
    matches: matches.slice(0, 5).map((match, index) => ({
      id: match.identity_id || match.id || `Candidate ${index + 1}`,
      score: clampScore(Number(match.score ?? match.confidence ?? 0)),
    })),
    reasons: payload?.reasons || payload?.reason_codes || [],
  };
}

function clampScore(value) {
  if (!Number.isFinite(value)) return 0;
  if (value > 1) return Math.max(0, Math.min(1, value / 100));
  return Math.max(0, Math.min(1, value));
}
