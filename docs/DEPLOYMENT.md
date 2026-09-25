# Deploying ClaimFlow

There are two ways to deploy ClaimFlow:
- **[Free deployment](#free-deployment)** costs nothing. The services sleep when idle.
- **[Render Blueprint](#paid-deployment-render-blueprint)** is paid and always on. Everything except Qdrant runs
  on Render.

For local development, `docker compose up -d` is all you need (see the README).

> ClaimFlow's recommendations are AI-assisted and require human review. A deployment is a demonstration system
> with synthetic data; do not load real customer data into it.

## Free deployment

The API needs about 600 MB of memory before it serves a request: torch plus the embedding model, measured on the
production image. Render's free instances have 512 MB, so the free route runs the API on Hugging Face, which gives
free Docker Spaces 16 GB.

```
Browser ──HTTPS + Basic auth──▶ Render free web service (console)
                                   │  adds X-API-Key server-side
                                   ▼  HTTPS
                                Hugging Face Space (API, Docker, 2 vCPU / 16 GB)
                                   ├──▶ Neon free Postgres
                                   ├──▶ Qdrant Cloud free cluster
                                   └──▶ Groq / Gemini
```

Put Neon and Qdrant in **US East (N. Virginia)**, next to Hugging Face's servers. Each triage makes dozens of
database and search calls, so the distance adds up.

Free services sleep when idle:
- The console wakes 30–60 s after 15 minutes idle.
- The API needs a minute or two after 48 hours idle.
- Neon wakes in about a second.

Open the site a minute before a demo.

### 1. Database: Neon

1. At https://neon.tech, create a project: Postgres 16, region **AWS US East (N. Virginia)**.
2. Under **Connect**, turn **Connection pooling off**. ClaimFlow creates its own read-only login and uses prepared
   statements, and both need a direct connection.
3. Copy the connection string. Paste it as it is; the API adapts `sslmode` and `channel_binding` for its driver.

### 2. Vectors: Qdrant Cloud

Create a free cluster in **N. Virginia**. Copy its URL (add `:6333` if it is missing) and create an API key.

### 3. API: Hugging Face Space

1. Create a free account at https://huggingface.co.
2. Create a token with **Write** access at https://huggingface.co/settings/tokens.
3. On your machine, from the repository root:

   ```bash
   pip install --upgrade huggingface_hub
   hf auth login                                                   # paste the token when asked
   python deploy/huggingface_space.py <your-username>/claimflow-api
   ```

   The script creates the Space if needed. It uploads what git has committed under `backend/`, so commit first.
   It prints the Space's URL.
4. In the Space, open **Settings → Variables and secrets** and add these secrets. Generate each random value with
   `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

   | Secret | Value |
   |---|---|
   | `DATABASE_URL` | the Neon connection string |
   | `AGENT_DB_USER` | `claimflow_agent` |
   | `AGENT_DB_PASSWORD` | a random value |
   | `QDRANT_URL` / `QDRANT_API_KEY` | from Qdrant Cloud |
   | `GROQ_API_KEY` / `GOOGLE_API_KEY` | at least one |
   | `API_KEY` | a random value; the console needs the same one |
   | `CORS_ORIGINS` | `[]` |
   | `SEED_DEMO_DATA` | `true` |
   | `MAX_CONCURRENT_RUNS` | `2` |

5. The Space restarts with the secrets. Its **Logs** tab shows the same startup sequence described for Render
   [below](#deploy-on-render), ending in `startup complete`. Check `https://<space-url>/api/health`.

To deploy a new version, commit and run the script again.

### 4. Console: Render free web service

1. In Render, choose **New → Web Service** and pick the GitHub repository.
2. Set **Language** to Docker, **Root Directory** to `frontend`, **Instance type** to Free, and **Region** to
   Virginia.
3. Under **Advanced**, set the **Health check path** to `/healthz`.
4. Add these environment variables:

   | Variable | Value |
   |---|---|
   | `BACKEND_URL` | the Space URL, e.g. `https://username-claimflow-api.hf.space` |
   | `API_KEY` | the same value as the Space's `API_KEY` |
   | `CONSOLE_USER` | `reviewer` |
   | `CONSOLE_PASSWORD` | a random value; this is what you sign in with |

5. Deploy, open the `.onrender.com` URL, and sign in.

## Paid deployment: Render Blueprint

Always-on hosting, with everything except Qdrant on Render. The notification, environment and troubleshooting sections after it apply to both routes.

### What runs where

```
Browser ──HTTPS + Basic auth──▶ claimflow-web (Next.js)          Render web service, 0.5c-512mb
                                   │  adds X-API-Key server-side
                                   ▼  private network, host:port
                                claimflow-api (FastAPI + agents)  Render web service, 1c-2g
                                   ├──▶ claimflow-db (Postgres 16)   Render Postgres, private only
                                   ├──▶ Qdrant Cloud (policy vectors) HTTPS + API key
                                   ├──▶ Groq / Gemini (LLMs)         HTTPS
                                   └──▶ n8n (notifications)          HTTPS, HMAC-signed
```

- **Browsers talk only to the console.** The API key never reaches a browser, and the API has no CORS origins.
- **The API runs as one process on purpose.** The triage runner and the live-trace event stream live in memory,
  so a second worker would not see the first one's runs. To handle more load, give it a bigger instance rather
  than more instances.
- **The database accepts no outside connections** (`ipAllowList: []`).

### Before you start

1. **Accounts:** a GitHub account with this repository pushed to it, and a Render account.
2. **Qdrant Cloud:**
   1. Create a free cluster at https://cloud.qdrant.io, in a region near Singapore if you can.
   2. Copy the cluster URL (`https://….cloud.qdrant.io:6333`).
   3. Create an API key.
3. **LLM keys:** at least one of these. With both, Gemini takes over when Groq fails.
   - Groq: https://console.groq.com/keys
   - Google AI Studio: https://aistudio.google.com/apikey

### Deploy on Render

1. In Render, choose **New → Blueprint**, select the repository, and confirm `render.yaml`.
2. Render asks for the values marked `sync: false`:

   | Variable | Value |
   |----------|-------|
   | `QDRANT_URL` | Your Qdrant Cloud cluster URL |
   | `QDRANT_API_KEY` | Your Qdrant Cloud API key |
   | `GROQ_API_KEY` | Groq key (or leave blank if you give a Google key) |
   | `GOOGLE_API_KEY` | Google AI Studio key (or leave blank if you give a Groq key) |
   | `N8N_WEBHOOK_URL` | Leave blank for now; see [Notifications](#notifications-n8n) |

3. Apply. Render creates the database, builds both images and starts them. The first API build takes about 10
   minutes, mostly for torch and the embedding model.
4. Watch the **claimflow-api** logs. A healthy first start shows, in order:
   1. The migration runs.
   2. `provision_readonly_role: 'claimflow_agent' can SELECT from customers, policies, premium_payments, claims and write nothing.`
   3. The seed data is loaded, and the three policy PDFs are ingested into Qdrant.
   4. `startup complete`, with `"triage_enabled": true`.
5. Find the console password: **claimflow-web → Environment → `CONSOLE_PASSWORD`**.
6. Open the claimflow-web URL and sign in as `reviewer` with that password. Then:
   1. Open **Claims**.
   2. Pick a *Submitted* claim.
   3. Click **Run triage**.

#### What `render.yaml` sets up for you

- **`DATABASE_URL`** comes from the database as `postgres://…`. The API rewrites it for the asyncpg driver.
- **The agents' read-only login** is built from three values: `AGENT_DB_USER`, `AGENT_DB_PASSWORD` (generated),
  and `DATABASE_URL`. From these the API derives `TOOLS_DATABASE_URL`, because `render.yaml` cannot join
  strings. `scripts/start.sh` creates the login on every deploy, with SELECT on four tables and nothing else. The
  API then checks at startup that the login really is read-only.
- **Generated secrets:**
  - `API_KEY` is generated and shared with the console via `fromService`.
  - `WEBHOOK_SECRET` is generated too.
  - The API refuses to start in production with an `API_KEY` shorter than 32 characters; generated values are 44.
- **`CONSOLE_PASSWORD`** is generated, which turns on HTTP Basic auth for the whole console. `/healthz` is exempt
  so Render's health check works.

#### If the database user may not create roles

Render's default database user can create roles, so the step above works there. On a provider where it cannot:

1. The deploy still succeeds. The log shows `read-only role provisioning failed` with the reason, and the
   agents stay disabled. They fail closed and never fall back to the owner login.
2. To fix it:
   1. Create the login yourself as an admin user.
   2. Grant it `SELECT` on `customers, policies, premium_payments, claims` only.
   3. Set `TOOLS_DATABASE_URL` to that login.
3. The privilege check at startup refuses the login if it can do anything more.

## Notifications (n8n)

The API sends 4 events: `claim.triaged`, `claim.decided`, `claim.overridden` and `claim.info_requested`. The
workflow in `n8n/claimflow-notifications.json`:

1. Checks each event's HMAC-SHA256 signature over the exact body bytes, in constant time.
2. Rejects stale events with 401.
3. Routes the event:
   - `claim.triaged` goes to one of three branches: escalated, ready for review, or failed.
   - The other 3 events each get their own branch.
4. Posts a Slack-compatible message to `NOTIFY_WEBHOOK_URL`.

**Locally:**

```bash
# in .env: WEBHOOK_SECRET=<16+ random characters>, N8N_WEBHOOK_URL=http://n8n:5678/webhook/claimflow,
#          NOTIFY_WEBHOOK_URL=<a Slack incoming-webhook URL, or blank>
docker compose --profile notifications up -d
```

- n8n imports and publishes the workflow on every start.
- The editor is at http://localhost:5678. Messages appear under **Executions** even without a Slack URL.

**For the Render deployment**, use any n8n that Render can reach over HTTPS (n8n Cloud, or your own):

1. Import `n8n/claimflow-notifications.json` and publish it.
2. Give n8n three environment variables:
   - `CLAIMFLOW_WEBHOOK_SECRET`: the `WEBHOOK_SECRET` value from **claimflow-api → Environment**.
   - `NOTIFY_WEBHOOK_URL`: your Slack incoming-webhook URL. For Discord, use the channel webhook URL with `/slack`
     added.
   - `NODE_FUNCTION_ALLOW_BUILTIN=crypto` and `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`. The signature check needs
     Node's crypto module and these two values.

   On n8n Cloud, which has no environment variables, paste the secret and URL into the workflow's
   *Verify signature* and *Post to Slack* nodes instead.
3. Set `N8N_WEBHOOK_URL` on claimflow-api to the workflow's production webhook URL (`…/webhook/claimflow`), and
   redeploy.

Delivery is best effort: an n8n outage is logged by the API and never blocks a triage run or a reviewer's
decision.

## Environment variable reference

### API (`backend`)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DATABASE_URL` | yes | – | Owner login. `postgres://`, `postgresql://` and `postgresql+asyncpg://` are accepted; `sslmode` becomes asyncpg's `ssl`. |
| `TOOLS_DATABASE_URL` | for agents | derived | The agents' SELECT-only login. |
| `AGENT_DB_USER`, `AGENT_DB_PASSWORD` | for agents | – | Used to derive `TOOLS_DATABASE_URL` from `DATABASE_URL` when it is not set. The password must be 12–128 characters of letters, digits and `_ - + / =`. |
| `TOOLS_STATEMENT_TIMEOUT_MS` | no | `5000` | Per-query limit for the agent tools. |
| `QDRANT_URL`, `QDRANT_API_KEY` | yes | `http://localhost:6333`, – | Vector store for the policy wording. |
| `GROQ_API_KEY`, `GOOGLE_API_KEY` | at least one, for agents | – | LLM providers: Groq primary, Gemini fallback. |
| `GROQ_MODEL`, `GEMINI_MODEL` | no | `openai/gpt-oss-120b`, `gemini-2.5-flash` | Model overrides. |
| `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` | no | `60`, `2` | Per-call timeout, and retries before the fallback provider takes over. |
| `API_KEY` | production | – | Required on every endpoint except `/api/health`. 32+ characters in production. |
| `ENVIRONMENT` | no | `development` (`production` in the runtime image) | Turns on the production checks. |
| `CORS_ORIGINS` | no | `["http://localhost:3000"]` | JSON list. `[]` when only the console (server-side) calls the API. |
| `MAX_CONCURRENT_RUNS` | no | `2` | Triage runs in parallel; each makes about 10 LLM calls. |
| `MAX_UPLOAD_BYTES`, `MAX_PDF_PAGES` | no | `10485760`, `50` | Upload limits. |
| `N8N_WEBHOOK_URL` | no | – | Where events are sent; unset means events are only logged. |
| `WEBHOOK_SECRET` | with n8n | – | HMAC key for the `X-ClaimFlow-Signature` header. |
| `SEED_DEMO_DATA` | no | `false` | Runtime image only: load the synthetic data set on start. |
| `LOG_LEVEL` | no | `INFO` | JSON logs on stdout. |
| `CHECKPOINT_PATH` | no | temp dir | LangGraph checkpoints (SQLite); losing them only affects runs in flight. |
| `MCP_AUTH_TOKEN` | MCP over HTTP | – | Bearer token, 32+ characters (see SECURITY.md). |
| `PORT` | no | `8000` | Set by Render. |

### Console (`frontend`)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `BACKEND_URL` | yes | `http://localhost:8000` | The API. A bare `host:port` (Render's `hostport`) gets `http://` added. |
| `API_KEY` | yes | – | Added to every proxied request, server-side only. |
| `CONSOLE_PASSWORD` | deployed | – | When set, the whole console needs HTTP Basic auth. |
| `CONSOLE_USER` | no | `reviewer` | Basic auth user name. |
| `PORT` | no | `3000` | Set by Render. |

### n8n

| Variable | Purpose |
|----------|---------|
| `CLAIMFLOW_WEBHOOK_SECRET` | Same value as the API's `WEBHOOK_SECRET`. With none set, every event is rejected. |
| `NOTIFY_WEBHOOK_URL` | Slack-compatible destination; blank means messages are only built. |
| `NODE_FUNCTION_ALLOW_BUILTIN=crypto` | Lets the Code node compute and compare the HMAC. |
| `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` | Lets the workflow read the two values above. Keep other secrets out of this n8n's environment. |
| `N8N_ENCRYPTION_KEY` | Hosted n8n only: its key for stored credentials; keep it stable. Locally, n8n generates and keeps its own. |

## Images

| Image | Base | Contents | Runs as |
|-------|------|----------|---------|
| `backend` (last stage, `runtime`) | `python:3.11-slim` | venv with runtime dependencies only (CPU torch), the embedding model, app code | uid 1000; the code is read-only to it |
| `backend` (`--target dev`, used by compose) | same | + pytest, ruff, tests, editable install, dependency auto-sync | uid 1000 |
| `frontend` | `node:20-alpine` | Next.js standalone server and static assets | `node` |

Build them locally the same way Render does:

```bash
docker build -t claimflow-api ./backend
docker build -t claimflow-web ./frontend
```

## Troubleshooting

Start with `python -m scripts.doctor` (in the API container, or on Render with the service's shell): it checks
every dependency and prints the fix. Common cases:

- **The API deploy fails its health check.** `/api/health` returns 503 until Postgres and Qdrant both answer.
  Check `QDRANT_URL`: it needs the `https://` scheme and the `:6333` port.
- **The API restarts with "Out of memory".** The instance is too small. Use `1c-2g` or larger.
- **"Run triage" answers 503 "triage unavailable".** No LLM key is set, or read-only provisioning failed. The
  API's startup log says which.
- **The console shows "API unreachable".** `BACKEND_URL` is wrong, or the API is still deploying. Private
  networking needs both services in the same region, on paid instances.
- **Every n8n execution ends in "bad signature".** The API's `WEBHOOK_SECRET` and n8n's
  `CLAIMFLOW_WEBHOOK_SECRET` differ. Copy the value again and restart n8n.
