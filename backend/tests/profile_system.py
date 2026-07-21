from __future__ import annotations

from io import BytesIO

from PIL import Image

from nexgen_engine.api.service import EngineService
from nexgen_engine.benchmarks.speed_benchmark import benchmark_latency_ms


def make_image() -> bytes:
    image = Image.new("RGB", (180, 180), (122, 132, 152))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def main() -> None:
    payload = make_image()
    service = EngineService(audit_path="runtime/profile_audit.jsonl")
    service.enroll(payload, "profile-subject", {"workspace": "profile"})
    stats = benchmark_latency_ms(lambda: service.identify(payload, operator_id="profile"), iterations=5)
    print(stats)


if __name__ == "__main__":
    main()
