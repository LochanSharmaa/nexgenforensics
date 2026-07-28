# NexGen iMATCH — frontend

React 19 + Vite. Two surfaces in one app:

- **Public site** (`src/components/`) — the marketing pages at `/`,
  `/products/*`, `/solutions/*`, `/resources/*`.
- **Investigator workspace** (`src/workspace/`) — the authenticated tool at
  `/workspace/*`.

## Running

```bash
npm install
```

```bash
npm run dev
```

Opens on <http://localhost:5173>. The dev server proxies `/api` to the backend on
port 8443 (see `vite.config.js`), so the browser stays same-origin and no CORS
grant is needed. Start the backend first — see the [root README](../README.md).

## Configuration

Copy `.env.example` to `.env.local`. In development leave `VITE_IMATCH_API_BASE`
empty to use the proxy. In production set it to the API's absolute `https://`
origin; the client refuses a non-HTTPS endpoint in a production build.

## Workspace routes

| Route | Purpose | Minimum role |
|---|---|---|
| `/workspace` | Case list and case creation | investigator |
| `/workspace/cases/:id` | Search history, adjudication, report export | investigator |
| `/workspace/search` | Probe upload, ranked candidates | investigator |
| `/workspace/verify` | 1:1 comparison, no gallery involved | investigator |
| `/workspace/enrol` | Add subjects to the gallery | supervisor |
| `/workspace/audit` | Audit trail and chain verification | investigator |

`RequireAuth` in `App.jsx` is a usability boundary, not a security boundary: it
decides what to render, nothing more. Authorization is enforced by the API on
every request, so removing it would make the UI confusing but would not expose a
single record.

## Two deliberate constraints

**The public product page does not run searches.** Biometric search needs an
authenticated operator, a stated lawful basis, and a tenant gallery, and every
run is audited — none of which an anonymous visitor can satisfy. The console on
`/products/imatch` shows values explicitly labelled as samples and links to the
real workspace. Wiring a live search in would mean either exposing an
unauthenticated biometric endpoint or fabricating a result and presenting it as
real.

**Similarity is shown as a raw cosine value, never as a percentage.** Rendering
0.62 as "62% confident" invites the reader to treat it as the probability that
two images are the same person, which it is not. The same reasoning drives the
banner that appears whenever the backend is running without a recognition model:
the scores look completely normal in that state and mean nothing.

## Layout

```
src/
  workspace/          Investigator workspace
    components/       EngineBanner, CandidateTable, ProbeReport, ImageDropZone
  components/         Public marketing site
  context/            AuthContext
  services/           apiClient.js (transport, refresh), imatchApi.js (endpoints)
```

Tokens live in `sessionStorage`, not `localStorage`: an investigator's session
should not outlive the browser tab.

## Build

```bash
npm run build
```

Outputs to `dist/`. `vercel.json` at the repo root configures SPA rewrites and
security headers, including a CSP that blocks inline scripts.
