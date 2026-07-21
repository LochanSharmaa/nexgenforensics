const LOCAL_IMATCH_API_URL = "http://localhost:8443/api/imatch/search";

export const imatchApiUrl =
  import.meta.env.VITE_IMATCH_API_URL?.trim() ||
  (import.meta.env.DEV ? LOCAL_IMATCH_API_URL : "");

export async function runImatchSearch({ file, mode, sourceUrl, checks }) {
  validateApiConfiguration();

  if (!file && !sourceUrl?.trim()) {
    throw new Error("Upload a photo or enter a secure image URL first.");
  }

  const requestPayload = {
    mode,
    purpose: "authorized_imatch_demo",
    lawful_use_reason: "User-submitted iMatch page analysis",
    checks,
    source_url: sourceUrl?.trim() || undefined,
    image_base64: file ? await fileToBase64(file) : undefined,
  };

  const response = await fetch(imatchApiUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestPayload),
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

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      resolve(result.includes(",") ? result.split(",").pop() : result);
    };
    reader.onerror = () => reject(new Error("Could not read selected image."));
    reader.readAsDataURL(file);
  });
}

function validateApiConfiguration() {
  if (!imatchApiUrl) {
    throw new Error("Secure iMatch API endpoint is not configured for this deployment.");
  }

  if (import.meta.env.PROD && !imatchApiUrl.startsWith("https://")) {
    throw new Error("Production iMatch API endpoint must use HTTPS.");
  }
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
