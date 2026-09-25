# n8n workflow

`claimflow-notifications.json` turns ClaimFlow's webhook events into chat messages.

```
Webhook (raw body) → Verify signature → Signed? ─no──▶ 401
                                           └─yes─▶ 202 → Route by event ─┬▶ escalated        ┐
                                                                         ├▶ ready for review │
                                                                         ├▶ triage failed    ├▶ Destination set? → Post to Slack
                                                                         ├▶ overridden       │
                                                                         ├▶ decided          │
                                                                         └▶ info requested   ┘
```

- **Verify signature:**
  - Recomputes `sha256=HMAC(CLAIMFLOW_WEBHOOK_SECRET, raw body)` and compares it with the `X-ClaimFlow-Signature`
    header in constant time.
  - Rejects events older than 5 minutes.
  - Rejects everything when no secret is configured.
- **The response goes out before routing**, so a slow Slack never holds up the API. The API doesn't wait for it
  either.
- **Messages are informational.** They describe what happened; no approval or rejection happens in n8n.

To run it locally, and for the environment variables it reads, see [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md#notifications-n8n).
Compose imports and publishes this file on every start, so edit the JSON, not the copy inside n8n. To keep a change
made in the n8n editor, run `n8n export:workflow --id=claimflowNotify1` and commit the export.
