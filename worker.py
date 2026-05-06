#!/usr/bin/env python3
"""
Rocket Alumni LLM Contact Extractor
Pulls website summaries from Supabase, calls Vast.ai vLLM (Gemma 2 9B),
extracts structured contact JSON, upserts to output table.
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Optional

import httpx
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_RAW_POOLER_URL", "postgres://...")
VAST_VLLM_URL = os.getenv("VAST_VLLM_URL", "PLACEHOLDER").rstrip("/")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "4"))
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-9b-it")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
INPUT_VIEW = os.getenv("INPUT_VIEW", "v_rocket_llm_input_2026_05_06")
OUTPUT_TABLE = os.getenv("OUTPUT_TABLE", "rocket_llm_contacts_2026_05_06")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
TEMPERATURE_INITIAL = float(os.getenv("TEMPERATURE", "0.1"))
HEALTHCHECK_RETRIES = int(os.getenv("HEALTHCHECK_RETRIES", "20"))
HEALTHCHECK_DELAY_SEC = int(os.getenv("HEALTHCHECK_DELAY_SEC", "30"))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    logger.info("SIGTERM received, graceful shutdown...")
    shutdown_requested = True


@dataclass
class ContactExtractionResult:
    domain_key: str
    school_name: str
    city: str
    state: str
    contacts: list
    generic_emails: list
    owner_name: str
    owner_title: str
    model_used: str
    extracted_at: str
    latency_ms: float
    raw_response: str
    retry_count: int
    extraction_status: str
    source_summary_chars: int


def get_db_connection():
    """Create a fresh DB connection (not pooled, per Railway guidelines)."""
    return psycopg2.connect(SUPABASE_URL)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def healthcheck_vllm():
    """Verify vLLM endpoint is live and model is loaded."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(f"{VAST_VLLM_URL}/v1/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_names = [m.get("id") for m in models]
            logger.info(f"vLLM healthy; models loaded: {model_names}")
            return True
        except Exception as e:
            logger.error(f"vLLM healthcheck failed: {e}")
            raise


async def wait_for_vllm_startup():
    """Poll vLLM until ready, up to 10 minutes."""
    deadline = time.time() + (HEALTHCHECK_RETRIES * HEALTHCHECK_DELAY_SEC)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            await healthcheck_vllm()
            logger.info(f"vLLM ready after {attempt} attempts")
            return True
        except Exception as e:
            remaining = deadline - time.time()
            logger.warning(
                f"Attempt {attempt}: vLLM not ready ({e}), "
                f"retrying in {HEALTHCHECK_DELAY_SEC}s ({remaining:.0f}s left)"
            )
            await asyncio.sleep(HEALTHCHECK_DELAY_SEC)
    logger.error("vLLM did not become ready within timeout")
    return False


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    if text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_prompt(row: dict) -> tuple[str, str]:
    """Build system and user prompt from row data."""
    system = (
        "You extract contact information from school website text. "
        "Output STRICT JSON only — no markdown, no explanations, no code fences."
    )
    user = f"""Website summary for {row['school_name']} ({row['city']}, {row['state']}):

{row['website_summary_8k']}

Extract every named person mentioned (administrators, board members, principals, athletic directors, alumni directors, coaches, teachers, staff). For each person extract their title and email if present in the text.

Return ONLY this JSON structure:
{{
  "contacts": [
    {{"name": "Jane Smith", "title": "Principal", "email": "jane@example.edu"}}
  ],
  "generic_emails": ["info@example.edu", "admissions@example.edu"],
  "owner_name": "John Smith",
  "owner_title": "Head of School"
}}

Rules:
- Only extract data EXPLICITLY present in the text. Never fabricate.
- contacts: array of {{name, title, email}}. Empty string for unknown email.
- generic_emails: department emails (info@, admissions@, contact@, etc).
- owner_name/owner_title: single top executive (Head of School, Superintendent, Principal) if clearly named. Otherwise empty string.
- Return valid JSON. No prose."""
    return system, user


async def call_vllm(system: str, user: str, temperature: float = TEMPERATURE_INITIAL) -> Optional[str]:
    """Call vLLM endpoint and return raw response text."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": temperature,
    }
    # Try with response_format; many vLLM instances ignore it gracefully
    try:
        payload["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VAST_VLLM_URL}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.error(f"vLLM call failed: {e}")
        return None


def parse_response(raw: str) -> Optional[dict]:
    """Parse JSON from response, strip markdown if needed."""
    if not raw:
        return None
    cleaned = strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}\nRaw: {cleaned[:200]}")
        return None


async def process_row(
    row: dict, semaphore: asyncio.Semaphore, client_http: httpx.AsyncClient
) -> ContactExtractionResult:
    """Process a single row: call vLLM, parse, return result."""
    async with semaphore:
        start = time.time()
        system, user = build_prompt(row)
        retry_count = 0
        temperature = TEMPERATURE_INITIAL
        parsed = None
        raw_response = ""

        # Retry up to 3 times with temperature bump
        for attempt in range(3):
            raw_response = await call_vllm(system, user, temperature)
            if raw_response:
                parsed = parse_response(raw_response)
                if parsed:
                    retry_count = attempt
                    break
            temperature = [0.1, 0.3, 0.5][attempt]

        latency_ms = (time.time() - start) * 1000

        # Build result
        if parsed:
            extraction_status = "success"
            contacts = parsed.get("contacts", [])
            generic_emails = parsed.get("generic_emails", [])
            owner_name = parsed.get("owner_name", "")
            owner_title = parsed.get("owner_title", "")
        else:
            extraction_status = "parse_failed"
            contacts = []
            generic_emails = []
            owner_name = ""
            owner_title = ""

        return ContactExtractionResult(
            domain_key=row["domain_key"],
            school_name=row["school_name"],
            city=row["city"],
            state=row["state"],
            contacts=contacts,
            generic_emails=generic_emails,
            owner_name=owner_name,
            owner_title=owner_title,
            model_used=MODEL_NAME,
            extracted_at=datetime.utcnow().isoformat(),
            latency_ms=latency_ms,
            raw_response=raw_response,
            retry_count=retry_count,
            extraction_status=extraction_status,
            source_summary_chars=len(row.get("website_summary_8k", "")),
        )


async def upsert_batch(results: list[ContactExtractionResult]):
    """Upsert results to output table."""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        for result in results:
            cur.execute(
                f"""
                INSERT INTO {OUTPUT_TABLE}
                (domain_key, school_name, city, state, contacts, generic_emails,
                 owner_name, owner_title, model_used, extracted_at, latency_ms,
                 raw_response, retry_count, extraction_status, source_summary_chars, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (domain_key) DO UPDATE SET
                  contacts = EXCLUDED.contacts,
                  generic_emails = EXCLUDED.generic_emails,
                  owner_name = EXCLUDED.owner_name,
                  owner_title = EXCLUDED.owner_title,
                  model_used = EXCLUDED.model_used,
                  extracted_at = EXCLUDED.extracted_at,
                  latency_ms = EXCLUDED.latency_ms,
                  raw_response = EXCLUDED.raw_response,
                  retry_count = EXCLUDED.retry_count,
                  extraction_status = EXCLUDED.extraction_status,
                  source_summary_chars = EXCLUDED.source_summary_chars,
                  updated_at = now()
                """,
                (
                    result.domain_key,
                    result.school_name,
                    result.city,
                    result.state,
                    extras.Json(result.contacts),
                    extras.Json(result.generic_emails),
                    result.owner_name,
                    result.owner_title,
                    result.model_used,
                    result.extracted_at,
                    result.latency_ms,
                    result.raw_response,
                    result.retry_count,
                    result.extraction_status,
                    result.source_summary_chars,
                ),
            )
        conn.commit()
        logger.info(f"Upserted {len(results)} rows")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def fetch_batch() -> list[dict]:
    """Fetch next batch from input view."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=extras.RealDictCursor)
        cur.execute(f"SELECT * FROM {INPUT_VIEW} LIMIT %s", (BATCH_SIZE,))
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Fetch batch failed: {e}")
        return []
    finally:
        cur.close()
        conn.close()


async def main():
    """Main worker loop."""
    global shutdown_requested

    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting Rocket Alumni LLM Contact Extractor")
    logger.info(f"  vLLM URL: {VAST_VLLM_URL}")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Batch size: {BATCH_SIZE}, Concurrency: {WORKER_CONCURRENCY}")

    # Healthcheck startup
    if not await wait_for_vllm_startup():
        logger.error("Failed to connect to vLLM; exiting")
        sys.exit(1)

    rows_done = 0
    empty_batch_count = 0
    start_time = time.time()
    semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)

    while not shutdown_requested:
        batch = fetch_batch()

        if not batch:
            empty_batch_count += 1
            if empty_batch_count >= 5:
                elapsed = time.time() - start_time
                rate = rows_done / elapsed if elapsed > 0 else 0
                logger.info(
                    f"5 consecutive empty batches; shutting down. "
                    f"Total: {rows_done} rows in {elapsed:.1f}s ({rate:.2f} rows/sec)"
                )
                break
            logger.info(f"Batch empty, sleeping 60s (empty count: {empty_batch_count}/5)")
            await asyncio.sleep(60)
            continue

        empty_batch_count = 0

        # Healthcheck per 100 rows
        if rows_done % 100 == 0 and rows_done > 0:
            try:
                await healthcheck_vllm()
            except Exception as e:
                logger.warning(f"Healthcheck failed at {rows_done} rows: {e}")

        # Process batch concurrently
        try:
            async with httpx.AsyncClient() as http_client:
                tasks = [process_row(row, semaphore, http_client) for row in batch]
                results = await asyncio.gather(*tasks, return_exceptions=False)

            # Filter out exceptions
            results = [r for r in results if isinstance(r, ContactExtractionResult)]

            # Upsert results
            await upsert_batch(results)

            rows_done += len(results)
            elapsed = time.time() - start_time
            rate = rows_done / elapsed if elapsed > 0 else 0
            failures = sum(1 for r in results if r.extraction_status != "success")

            if rows_done % 50 == 0:
                logger.info(
                    f"Progress: {rows_done} rows, rate={rate:.2f}/sec, "
                    f"failures={failures}/{len(results)}"
                )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

    elapsed = time.time() - start_time
    rate = rows_done / elapsed if elapsed > 0 else 0
    logger.info(f"Worker done: {rows_done} rows in {elapsed:.1f}s ({rate:.2f} rows/sec)")


if __name__ == "__main__":
    asyncio.run(main())
