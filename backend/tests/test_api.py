from __future__ import annotations

from io import BytesIO
import base64

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_image() -> bytes:
    image = Image.new("RGB", (180, 180), (110, 130, 150))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def test_health_and_engine_routes():
    client = TestClient(app)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    payload = make_image()
    enroll = client.post(
        "/api/v1/engine/enroll",
        json={"identity_id": "api-subject", "workspace": "api", "image_base64": base64.b64encode(payload).decode("ascii")},
    )
    assert enroll.status_code == 200
    identify = client.post(
        "/api/v1/engine/identify",
        json={"operator_id": "api-tester", "image_base64": base64.b64encode(payload).decode("ascii")},
    )
    assert identify.status_code == 200
    assert identify.json()["matches"]
