# Deploying ClaimFlow

ClaimFlow deploys for free:
- **[Render free plan](#free-deployment-on-render)** runs the API and the console from `render.yaml`, with Neon
  Postgres and a Qdrant Cloud cluster, both free.
- **[Hugging Face Space](#alternative-the-api-on-a-hugging-face-space)** can run the API instead. It has far more
  CPU, so it starts faster.

For local development, `docker compose up -d` is all you need (see the README).

> ClaimFlow's recommendations are AI-assisted and require human review. A deployment is a demonstration system
> with synthetic data; do not load real customer data into it.

## Free deployment on Render

```
Browser ──HTTPS + Basic auth──▶ claimflow-web (Next.js)          Render free web service, 512 MB
                                   │  adds X-API-Key server-side
                                   ▼  HTTPS (BACKEND_URL)
                                claimflow-api (FastAPI + agents)  Render free web service, 512 MB, 0.1 CPU
                                   ├──▶ Neon free Postgres
                                   ├──▶ Qdrant Cloud free cluster (policy vectors)
                                   ├──▶ Groq / Gemini (LLMs)
                                   └──▶ n8n (notifications, optional)
```

- **Browsers talk only to the console.** The API key never reaches a browser, and the API has no CORS origins.
- **The console calls the API over HTTPS.** Free services cannot receive private network traffic.
- **The API runs as one process on purpose.** The triage runner and the live-trace event stream live in memory,
  so a second worker would not see the first one's runs.

### Memory and start-up time

The API fits a free instance because production runs the embedding model's ONNX export with onnxruntime, without
torch (torch alone took about 400 MB). Measured on the production image, limited to 512 MB and 0.1 CPU:

| | Memory | Time to answer |
|---|---|---|
| After start-up | ~345 MB | about 3 minutes on the first deploy, which embeds the three policy PDFs once |
| Waking after 15 idle minutes | ~345 MB | about 70 seconds, plus the time Render takes to start the instance |
| Two triage runs at once, or a 10 MB upload | ~375 MB at most | |

All of start-up runs in one Python process (`scripts/serve.py`), so the application is imported and the model
loaded once. On a tenth of a CPU, each extra process cost 15–20 seconds.

Free instances sleep after 15 minutes without traffic. **Open the console a few minutes before a demo**, then run
one triage to wake Neon and Qdrant too.

### Before you start

1. **Accounts:** GitHub with this repository pushed to it, and Render (sign in with GitHub).
2. **LLM keys:** at least one. With both, Gemini takes over when Groq fails. Free keys have daily limits (see
   the README's Limitations).
   - Groq: https://console.groq.com/keys
   - Google AI Studio: https://aistudio.google.com/apikey

### 1. Database: Neon

1. At https://neon.tech, create a project: Postgres 16, region **AWS Asia Pacific 1 (Singapore)**, next to the
   Render services. Each triage makes dozens of database calls, so distance adds up.
2. Under **Connect**, turn **Connection pooling off**. ClaimFlow creates its own read-only login and uses prepared
   statements, and both need a direct connection.
3. Copy the connection string. Paste it as it is; the API adapts `sslmode` and `channel_binding` for its driver.

### 2. Vectors: Qdrant Cloud

1. At https://cloud.qdrant.io, create a free cluster, in the region nearest Singapore offered.
2. Copy its URL (`https://….cloud.qdrant.io`; add `:6333` if it is missing) and create an API key.

### 3. Deploy the Blueprint

1. In Render, choose **New → Blueprint**, select the repository, and confirm `render.yaml`.
2. Render asks for the values marked `sync: false`:

   | Variable | Value |
   |----------|-------|
   | `DATABASE_URL` | The Neon connection string |
   | `QDRANT_URL` / `QDRANT_API_KEY` | From Qdrant Cloud |
   | `GROQ_API_KEY` / `GOOGLE_API_KEY` | At least one; leave the other blank |
   | `N8N_WEBHOOK_URL` | Leave blank for now; see [Notifications](#notifications-n8n) |
   | `BACKEND_URL` (claimflow-web) | `https://claimflow-api.onrender.com`. If Render gives the API another URL (the name was taken), correct it afterwards under **claimflow-web → Environment**. |

3. Apply. Render builds both images and starts them. The first API build takes about 5 minutes.
4. Watch the **claimflow-api** logs. A healthy first start shows, in order:
   1. `migrate: schema is at the latest migration.`
   2. `provision_readonly_role: 'claimflow_agent' can SELECT from customers, policies, premium_payments, claims and write nothing.`
   3. `Seeded 50 customers, 80 policies, …`, then three `ingest: …: ingested` lines.
   4. `startup complete`, with `"triage_enabled": true`.
5. Check that the API's URL (top of its page) matches the console's `BACKEND_URL`.
6. Find the console password: **claimflow-web → Environment → `CONSOLE_PASSWORD`**.
7. Open the claimflow-web URL and sign in as `reviewer` with that password. Then:
   1. Open **Claims**.
   2. Pick a *Submitted* claim.
   3. Click **Run triage**.

For always-on hosting without the wait, change both `plan: free` in `render.yaml` to `starter` or larger.

#### What `render.yaml` sets up for you

- **The agents' read-only login** is built from three values: `AGENT_DB_USER`, `AGENT_DB_PASSWORD` (generated),
  and `DATABASE_URL`. From these the API derives `TOOLS_DATABASE_URL`, because `render.yaml` cannot join
  strings. `scripts/serve.py` creates the login on every start, with SELECT on four tables and nothing else. The
  API then checks at startup that the login really is read-only.
- **Generated secrets:**
  - `API_KEY` is generated and shared with the console via `fromService`.
  - `WEBHOOK_SECRET` is generated too.
  - The API refuses to start in production with an `API_KEY` shorter than 32 characters; generated values are 44.
- **`CONSOLE_PASSWORD`** is generated, which turns on HTTP Basic auth for the whole console. `/healthz` is exempt
  so Render's health check works.
- **`SEED_DEMO_DATA=true`** loads the synthetic data set. Production refuses to seed without it, and never wipes
  data the app has written.

#### If the database user may not create roles

Neon's project owner can create roles, so the step above works there. On a provider where it cannot:

1. The deploy still succeeds. The log shows `read-only role provisioning failed` with the reason, and the
   agents stay disabled. They fail closed and never fall back to the owner login.
2. To fix it:
   1. Create the login yourself as an admin user.
   2. Grant it `SELECT` on `customers, policies, premium_payments, claims` only.
   3. Set `TOOLS_DATABASE_URL` to that login.
3. The privilege check at startup refuses the login if it can do anything more.

## Alternative: the API on a Hugging Face Space

A free Docker Space has 2 vCPU and 16 GB, so the API starts in seconds instead of minutes. It sleeps after 48
hours without traffic. Use the same Neon and Qdrant setup, ideally in **US East (N. Virginia)**, next to Hugging
Face's servers.

1. Create a free account at https://huggingface.co, and a token with **Write** access at
   https://huggingface.co/settings/tokens.
2. On your machine, from the repository root:

   ```bash
   pip install --upgrade huggingface_hub
   hf auth login                                                   # paste the token when asked
   python deploy/huggingface_space.py <your-username>/claimflow-api
   ```

   The script creates the Space if needed. It uploads what git has committed under `backend/`, so commit first.
   It prints the Space's URL.
3. In the Space, open **Settings → Variables and secrets** and add these secrets. Generate each random value with
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

4. The Space restarts with the secrets. Its **Logs** tab shows the start-up sequence above, ending in
   `startup complete`. Check `https://<space-url>/api/health`.
5. Run the console on Render as above, with `BACKEND_URL` set to the Space URL (for example
   `https://username-claimflow-api.hf.space`) and `API_KEY` to the Space's value; or delete claimflow-api from
   `render.yaml` first.

To deploy a new version, commit and run the script again.

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
| `SEED_DEMO_DATA` | no | `false` | Runtime image only: load the synthetic data set on start. The only way to seed with `ENVIRONMENT=production`. |
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
| `backend` (last stage, `runtime`) | `python:3.11-slim` | venv with runtime dependencies only (onnxruntime, no torch), the embedding model's ONNX export, app code | uid 1000; the code is read-only to it |
| `backend` (`--target dev`, used by compose) | same | + CPU torch, sentence-transformers, pytest, ruff, tests, editable install, dependency auto-sync | uid 1000 |
| `frontend` | `node:20-alpine` | Next.js standalone server and static assets | `node` |

Build them locally the same way Render does, and try the API under the free plan's limits:

```bash
docker build -t claimflow-api ./backend
docker build -t claimflow-web ./frontend
docker run --memory 512m --cpus 0.1 --env-file <your production env> -p 8000:8000 claimflow-api
```

## Troubleshooting

Start with `python -m scripts.doctor` (in the API container; free Render services have no shell, so run it
locally against the same `DATABASE_URL` and `QDRANT_URL`): it checks every dependency and prints the fix. Common
cases:

- **The API deploy fails its health check.** `/api/health` returns 503 until Postgres and Qdrant both answer.
  Check `QDRANT_URL`: it needs the `https://` scheme and the `:6333` port.
- **The API restarts with "Out of memory".** It peaks around 375 MB with `MAX_CONCURRENT_RUNS=2`; raising that
  adds memory per run. Lower it, or use a larger instance.
- **"Run triage" answers 503 "triage unavailable".** No LLM key is set, or read-only provisioning failed. The
  API's startup log says which.
- **A triage ends "escalated" with confidence 0.** Every LLM provider refused, usually a free key's daily quota.
  The API's log shows `LLM provider out of daily quota`. The run fails safe to a human; try again later.
- **The console shows "API unreachable".** `BACKEND_URL` must be the API's full `https://…onrender.com` URL. Right
  after the console wakes, the API may still be starting; reload after a minute.
- **Every n8n execution ends in "bad signature".** The API's `WEBHOOK_SECRET` and n8n's
  `CLAIMFLOW_WEBHOOK_SECRET` differ. Copy the value again and restart n8n.
