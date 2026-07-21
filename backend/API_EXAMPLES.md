# API Examples

The backend accepts JSON with base64 image payloads, so it can run even when multipart upload support is not installed.

## Health

```http
GET /api/v1/health
```

## Enroll

```json
{
  "identity_id": "customer-001",
  "workspace": "tenant-a",
  "image_base64": "<base64 image>"
}
```

## Identify

```json
{
  "operator_id": "analyst-001",
  "image_base64": "<base64 image>"
}
```

## iMatch Search

```json
{
  "mode": "single",
  "purpose": "authorized_imatch_demo",
  "lawful_use_reason": "consent-approved verification",
  "checks": ["Liveness Check", "Quality Assessment"],
  "image_base64": "<base64 image>"
}
```

Multipart form uploads are also supported when `python-multipart` is installed.
