# NexGen Forensics

Project structure:

- `frontend` - Vite React website deployed to Vercel.
- `backend` - reserved for API/server code.

## Frontend

```sh
cd frontend
npm install
npm run dev
```

Production build:

```sh
cd frontend
npm run build
```

## Production Security

- Do not commit `.env` files, tokens, private keys, database dumps, or deployment credentials.
- Configure production environment variables inside Vercel project settings.
- `VITE_IMATCH_API_URL` must use an HTTPS endpoint in production.
- The Vercel deployment includes security headers for HTTPS, framing protection, content sniffing protection, referrer policy, permissions policy, and content security policy.

## Backend

The backend folder is ready for server code when needed.
