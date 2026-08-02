# NexGen iMATCH web tier: build the SPA, serve it with nginx, and proxy /api
# to the backend container — ONE origin for the browser.
#
# Build from the frontend/ directory (docker-compose.selfhost.yml does this):
#   docker build -t nexgen-imatch-web .
#
# Same-origin is the point of this image. The bundle is built with
# VITE_IMATCH_API_BASE=same-origin, so every API call is a relative /api path
# answered by this nginx and forwarded to the api service. No CORS grant, no
# cross-site cookies, no CSRF cookie rejection — the failure modes that broke
# login on split hosting (Vercel + Render) cannot occur here.

FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# Overrides the committed .env.production (which names the old Render origin):
# Vite gives real environment variables precedence over .env files.
ENV VITE_IMATCH_API_BASE=same-origin
RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
