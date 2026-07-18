from __future__ import annotations

from app.schemas import ForensicFindings

STANDARD_SCOPE_TEXT = (
    "This report reflects automated facial similarity analysis performed by the NexGen system at [config_version]. "
    "Automated comparison results constitute an investigative lead and are distinct from human forensic morphological examination. "
    "Findings are subject to review by a qualified examiner prior to evidentiary or operational use."
)
QUALITY_THRESHOLD = 0.60
LIVENESS_THRESHOLD = 0.75
POSE_THRESHOLD_DEGREES = 15.0


def build_narrative(findings: ForensicFindings) -> dict[str, str]:
    """Create reproducible report prose without network or model calls."""
    methodology = (
        f"{len(findings.model_scores)} comparison model(s) were used under configuration "
        f"{findings.config_version}; decision threshold: {findings.threshold_value:.3f}."
    )
    observations: list[str] = []
    for number, image in enumerate(findings.source_images, start=1):
        image_notes = [
            f"Image {number}: quality score {image.quality_score:.2f}",
            f"yaw {image.pose_yaw:.1f}°, pitch {image.pose_pitch:.1f}°",
            f"liveness score {image.liveness_score:.2f}",
        ]
        flags: list[str] = []
        if image.quality_score < QUALITY_THRESHOLD:
            flags.append("quality is below the accepted threshold")
        if abs(image.pose_yaw) > POSE_THRESHOLD_DEGREES or abs(image.pose_pitch) > POSE_THRESHOLD_DEGREES:
            flags.append("pose angle exceeds 15 degrees")
        if image.liveness_score < LIVENESS_THRESHOLD:
            flags.append("liveness is below the accepted threshold")
        observations.append("; ".join(image_notes) + (f" — limiting factor: {', '.join(flags)}." if flags else "."))
    has_limitations = any("limiting factor" in note for note in observations)
    prefix = "Limiting factors requiring examiner review: " if has_limitations else "No material quality or pose limitations were identified. "
    return {
        "methodology_summary": methodology,
        "quality_notes": prefix + " ".join(observations),
        "scope_statement": STANDARD_SCOPE_TEXT.replace("[config_version]", findings.config_version),
    }
