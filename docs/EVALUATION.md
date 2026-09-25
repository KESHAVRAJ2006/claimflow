# Evaluation

ClaimFlow is evaluated at three levels. Each level has its own labelled data and its own script, and each script
exits non-zero when it misses its target.

| Level | What is checked | Data | Command |
|---|---|---|---|
| Retrieval | Does the right policy page come back for a question? | 12 questions labelled with the pages that answer them, plus off-topic questions | `python -m scripts.eval_retrieval` |
| Rules | Do R01–R08 fire exactly when they should? | The seed's 8 edge-case claims (one per rule) and every other seeded claim, which must trigger none | `pytest tests/test_rules_on_seed.py` |
| End-to-end triage | Is the recommendation right, safe and grounded? | 20 labelled claims (below) | `python -m scripts.eval_triage` |

Run the commands inside the backend container (`docker compose exec backend ...`).

## Retrieval

Measured on the three policy PDFs with the real embedding model, in an in-memory Qdrant, so the shared index cannot
affect the result.

| Metric | With product filter | Without |
|---|---|---|
| Recall@5 | **1.00** (12/12) | 1.00 |
| MRR | 0.86 | 0.78 |

- For answerable questions, retrieval confidence has a median of 0.96 and a minimum of 0.66.
- Off-topic questions score 0.00.
- Routing escalates below 0.65, so an off-topic "citation" can never support an approval.

## End-to-end triage

### The labelled set

[`backend/data/eval/triage_cases.json`](../backend/data/eval/triage_cases.json) has 20 cases.

- **12 coverage cases:** the policy wording decides. Each one borrows a clean seeded claim (policy, customer,
  dates and amount, with no rule triggered) and replaces its description with a story that turns on one clause.
- **8 rule cases:** the seeded edge cases R01–R08, where a deterministic rule decides.

Every label records:
- the expected outcome;
- any defensible alternatives (usually escalation: sending a claim to a person is slower, never unsafe);
- the expected coverage judgment;
- the clause the decision should cite;
- the reasoning.

| Case | Story | Label | Deciding clause |
|---|---|---|---|
| M1 | Rear-ended at a traffic signal | approve | Motor 3.1 own damage |
| M2 | Car stolen with the keys in the ignition | reject | Motor 4.4 unlocked or unattended vehicle |
| M3 | Engine seized after restarting in floodwater | reject | Motor 4.3 hydrostatic lock |
| M4 | Crash with alcohol above the legal limit | reject | Motor 4.1 general exclusions |
| M5 | Car stolen overnight, nothing said about locks or keys | approve | Motor 3.1 (the locks are a question for the reviewer) |
| H1 | Wrist fracture from a fall, 3 days in hospital | approve | Health 3.6 accidental injuries |
| H2 | Cataract surgery in the first 24 months | reject | Health 5.2 specific waiting periods |
| H3 | Liposuction for weight reduction | reject | Health 6.1 treatments not covered |
| H4 | Dengue admission | approve | Health 3.1 inpatient hospitalisation |
| HM1 | Pipe burst suddenly | approve | Home 3.3 bursting of pipes |
| HM2 | Damp patches from a slow roof leak | reject | Home 6.2 gradual water damage |
| HM3 | Jewellery missing, no forced entry | reject | Home 6.4 theft without forcible entry |
| R01 | Claimed 650,000 on a 500,000 sum insured | escalate | Rule R01 (risk 70; above the auto-decision limit) |
| R02 | Incident after the policy lapsed | reject | Rule R02 hard block |
| R03 | Cyclone damage 12 days after cover began | approve | Rule R03 alone scores 40, below the threshold |
| R04 | Fourth claim in a year, burst pipe | approve | Rule R04 alone scores 35 |
| R05 | Incident dated in the future | reject | Rule R05 hard block |
| R06 | Incident before cover began | reject | Rule R06 hard block |
| R07 | Duplicate of an approved claim | escalate | Rule R07 (risk 70) |
| R08 | KYC pending | escalate | Rule R08 (risk 70) |

**How a case runs:**
- It uses the real graph, with the real LLM, tools, retrieval and rules.
- The run happens in memory, without a checkpointer.
- The agent tools use the read-only database login. **Nothing is written**, so the claims in the console are never
  touched.
- Results are appended to `backend/data/eval/triage_results.jsonl` as each case finishes.
- A later run skips cases already scored. `--report` rebuilds the report from saved results without calling a model.

### Metrics

In order of importance:

1. **Unsafe approvals:** the recommendation is *approve* where the label allows only reject or escalate. The target
   is **0**. Every design decision (hard blocks, the routing table, the weakest-citation confidence) exists to keep
   this at zero, so it is the first line of the report, and the script exits 1 if it is not zero.
2. **Outcome accuracy:** measured two ways, as an exact match with the label and as a match with the label or a
   defensible alternative.
3. **Wrongful rejections:** *reject* where the label says approve.
4. **Coverage judgment** and **deciding clause cited:** whether the decision agent read the policy correctly, and
   pointed the reviewer at the clause that settles it.
5. **Escalation rate:** how much work reaches a senior reviewer, and how much of it the label did not require.
6. **Reflection retries** and **wasted retries:** a retry is wasted when its investigation round makes no new tool
   call and finds no new passage.
7. **Latency, LLM calls and tokens per claim:** these give the free-tier capacity.

Runs where a provider refused a call (quota or outage) are recorded but **kept out of the metrics**. They measure
the provider, not the pipeline, and are re-run on the next invocation.

### Results

Generated into [`backend/data/eval/triage_report.md`](../backend/data/eval/triage_report.md) by
`python -m scripts.eval_triage --report`.

> **Status (25 Sep 2026): not yet scored.**
> - The harness and all its tests are complete.
> - The first live runs used up both free daily quotas: Groq's rolling daily token limit and Gemini's 20 requests
>   a day.
> - The measurement below is why.
> - Running `python -m scripts.eval_triage` once a day continues where it stopped.

**Measured cost of one claim** (case M5, one of the first runs):
- 9 LLM calls, 32,398 tokens (29,445 in, 2,953 out) and 162 s.
- That run was still cut short by a quota, so a complete claim costs more.
- Groq's free tier allows 8,000 tokens a minute. One claim therefore needs about 4 minutes of token budget.
- Only a handful of claims fit in a day's token allowance, which makes the full set of 20 a job of several days on
  free keys.

## Reflection tuning

**Before.** The reflection agent's LLM critic could send a claim back for more investigation whenever it wanted a
fact. The development database held 5 real runs from before this phase:

| Claim | What the critic asked for | Could a tool answer it? |
|---|---|---|
| CLM-2026-000186 (theft) | whether the vehicle was locked, or keys were left inside | No: only the claimant knows |
| CLM-2026-000186, again | the same, reworded | No |
| CLM-2026-900003 | "claim form details showing the loss pertains to roof and walls" | No: the form was already in the prompt |
| CLM-2026-900003, again | the same, reworded | No |
| CLM-2026-000192 | a citation with no supporting quote (found by code) | Yes: a real grounding problem |

So 4 of the 5 retries could not succeed. Both claims then ran out of retries and escalated. The pipeline spent the
most tokens on exactly the claims where more investigation could not help.

**Change.**
- The critic now sees the investigator's 9 tools. For each lookup it wants, it must name the tool that can answer it,
  or `"none"` when only the claimant, a surveyor or a missing document could.
- Code, not the model, then decides:
  - A lookup naming a real tool that has not been asked for before → send the claim back (as before, at most twice).
  - A lookup naming `"none"`, an unknown tool, or something already sent back once (rewording included) → an **open
    question for the reviewer**. It is shown on the decision card under "Check with the claimant", and added to the
    stored rationale.
  - Not grounded, and nothing a tool can fetch → escalate at once instead of burning retries.

The new behaviour is covered by tests in `tests/test_reflection.py` and `tests/test_graph.py`. Those tests replay
the two real failures above: the locked-car question and the reworded roof-and-walls request. The live effect is
measured by the *wasted retries* metric once the evaluation has run.

## Problems the evaluation found

1. **Seed data gave a motor claim a house's description.**
   - Natural calamity is both a motor and a home incident, but descriptions were chosen by incident type alone.
   - So CLM-2026-900003, a motor claim, read *"Roof and walls damaged by a falling tree"*, and the critic kept
     asking about roof and walls.
   - Descriptions are now keyed by product and incident. A test proves that every other seeded value (dates,
     amounts, IDs) is unchanged.
2. **Two tests read the real environment.**
   - Once notifications were enabled in `.env`, a doctor test sent a real signed ping to the running n8n.
   - The test configuration now blanks the n8n settings, as it already did for the LLM keys.
3. **Token cost is the binding constraint**, not accuracy or latency. See *Limitations* in the README.

## Reproducing

```bash
docker compose exec backend python -m scripts.eval_retrieval          # retrieval, no LLM
docker compose exec backend pytest tests/test_rules_on_seed.py        # rules, no LLM
docker compose exec backend python -m scripts.eval_triage --list      # which seeded claim each case uses
docker compose exec backend python -m scripts.eval_triage             # run unscored cases (resumes)
docker compose exec backend python -m scripts.eval_triage --only M2   # re-run one case
docker compose exec backend python -m scripts.eval_triage --report    # rebuild the report, no LLM
```
