# Rocket Alumni LLM Contact Extractor

Pulls school website summaries from a Supabase view, calls Vast.ai vLLM serving Gemma 2 9B, extracts structured contact information (administrator names, titles, emails, generic department emails) via JSON-mode prompting, and upserts results to a Supabase output table. Handles markdown-wrapped responses, retries with temperature escalation on parse failures, and performs periodic healthchecks. Designed for Railway deployment with async httpx and fresh DB connections per operation.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SUPABASE_RAW_POOLER_URL` | required | Postgres DSN (TX pooler 6543 preferred) |
| `VAST_VLLM_URL` | `PLACEHOLDER` | vLLM endpoint, e.g. `http://203.0.113.42:34590` |
| `BATCH_SIZE` | 50 | Rows fetched per iteration |
| `WORKER_CONCURRENCY` | 4 | Async tasks in parallel to vLLM |
| `MODEL_NAME` | `google/gemma-2-9b-it` | Model served by vLLM |
| `MAX_OUTPUT_TOKENS` | 1024 | Max tokens per completion |
| `TEMPERATURE` | 0.1 | Initial temperature (bumps to 0.3, 0.5 on retry) |
| `LOG_LEVEL` | `INFO` | Python log level |
| `INPUT_VIEW` | `v_rocket_llm_input_2026_05_06` | Source view name |
| `OUTPUT_TABLE` | `rocket_llm_contacts_2026_05_06` | Target table name |
| `HEALTHCHECK_RETRIES` | 20 | Startup vLLM healthcheck attempts |
| `HEALTHCHECK_DELAY_SEC` | 30 | Delay between healthcheck attempts |

## Features

- **Async pipeline** with httpx and psycopg2
- **Markdown fence stripping** for Gemma 2 9B response wrapper removal
- **Retry logic** with temperature escalation (0.1 → 0.3 → 0.5)
- **Idempotent upserts** via ON CONFLICT on domain_key
- **Periodic healthchecks** every 100 rows and at startup
- **Graceful shutdown** on SIGTERM (Railway)
- **Fresh connections** per batch (no pooling to avoid Railway exhaustion)
- **Continuous operation** with 5-empty-batch exit heuristic

## Deployment

```bash
railway up
```

Provide `SUPABASE_RAW_POOLER_URL` and `VAST_VLLM_URL` at service creation time.
