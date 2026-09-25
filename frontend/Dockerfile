# ClaimFlow console image, multi-stage.
#
#   deps     node_modules from the lockfile (cached until package-lock.json changes)
#   build    `next build` with output: "standalone" (lint and type checks run here and fail the build)
#   runtime  node + server.js + only the modules the server imports; no source, no dev dependencies

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
# ci: exactly the lockfile's versions, and fails if package.json and the lockfile disagree.
RUN npm ci --no-audit --no-fund


FROM node:20-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build


FROM node:20-alpine AS runtime
WORKDIR /app
# HOSTNAME=0.0.0.0: the standalone server otherwise binds to the container's hostname only. BACKEND_URL, API_KEY
# and CONSOLE_PASSWORD are read at request time, so one image serves every environment.
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# The node image ships an unprivileged "node" user. It owns the files because Next writes its cache under .next.
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
USER node

EXPOSE 3000
# /healthz is exempt from the console's Basic auth; busybox wget ships with alpine.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD wget -q -O /dev/null "http://127.0.0.1:${PORT}/healthz" || exit 1
CMD ["node", "server.js"]
