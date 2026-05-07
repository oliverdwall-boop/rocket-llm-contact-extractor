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
# VAST_VLLM_URL accepts comma-separated list for multi-GPU round-robin load balancing
_RAW_URLS = os.getenv("VAST_VLLM_URL", "PLACEHOLDER")
VAST_VLLM_URLS = [u.strip().rstrip("/") for u in _RAW_URLS.split(",") if u.strip()]
VAST_VLLM_URL = VAST_VLLM_URLS[0] if VAST_VLLM_URLS else "PLACEHOLDER"  # back-compat single-URL refs
import itertools as _itertools, threading as _threading
_url_cycle = _itertools.cycle(VAST_VLLM_URLS) if VAST_VLLM_URLS else _itertools.cycle(["PLACEHOLDER"])
_url_lock = _threading.Lock()
def _next_vllm_url() -> str:
    with _url_lock:
        return next(_url_cycle)
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
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
    """Verify EVERY configured LLM endpoint is live and model is loaded."""
    headers = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        for base in VAST_VLLM_URLS:
            base = base.rstrip("/")
            models_path = "/models" if base.endswith("/v1") else "/v1/models"
            try:
                resp = await client.get(f"{base}{models_path}", headers=headers)
                resp.raise_for_status()
                models = resp.json().get("data", [])
                model_names = [m.get("id") for m in models]
                logger.info(f"LLM healthy at {base}; models: {model_names[:3]}")
            except Exception as e:
                logger.error(f"LLM healthcheck failed for {base}: {e}")
                raise
        logger.info(f"All {len(VAST_VLLM_URLS)} vLLM endpoint(s) healthy")
        return True


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

    headers = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    # Round-robin: pick the next endpoint from the configured pool
    base = _next_vllm_url().rstrip("/")
    completions_path = "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"

    # Retry on 429 with backoff. Up to 5 attempts, exponentially increasing wait.
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base}{completions_path}",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 429:
                    # Honor Retry-After header if present, else exponential backoff
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait_s = float(retry_after)
                        except ValueError:
                            wait_s = 5 * (2 ** attempt)
                    else:
                        wait_s = 5 * (2 ** attempt)
                    wait_s = min(wait_s, 60)
                    logger.warning(f"429 rate-limited, sleeping {wait_s:.1f}s (attempt {attempt+1}/5)")
                    await asyncio.sleep(wait_s)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error: {e.response.status_code} {e.response.text[:200]}")
            return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    logger.error("LLM call gave up after 5 retries on 429")
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


def upsert_batch_sync(results: list[ContactExtractionResult]):
    """Sync upsert (called via asyncio.to_thread to keep event loop responsive)."""
    if not results:
        return
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        values = [
            (
                r.domain_key, r.school_name, r.city, r.state,
                extras.Json(r.contacts), extras.Json(r.generic_emails),
                r.owner_name, r.owner_title, r.model_used, r.extracted_at,
                r.latency_ms, r.raw_response, r.retry_count, r.extraction_status,
                r.source_summary_chars,
            )
            for r in results
        ]
        extras.execute_values(
            cur,
            f"""
            INSERT INTO {OUTPUT_TABLE}
            (domain_key, school_name, city, state, contacts, generic_emails,
             owner_name, owner_title, model_used, extracted_at, latency_ms,
             raw_response, retry_count, extraction_status, source_summary_chars)
            VALUES %s
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
            values,
            page_size=100,
        )
        conn.commit()
        logger.info(f"Upserted {len(results)} rows (execute_values)")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()


def fetch_batch() -> list[dict]:
    """Fetch next batch from input view with retry on transient pool errors."""
    last_err = None
    for attempt in range(5):
        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            cur.execute(
                f"SELECT * FROM {INPUT_VIEW} ORDER BY random() LIMIT %s",
                (BATCH_SIZE,),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            last_err = e
            logger.warning(f"Fetch batch attempt {attempt+1}/5 failed: {e}")
            try:
                if cur: cur.close()
            except Exception: pass
            try:
                if conn: conn.close()
            except Exception: pass
            time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16s backoff
    logger.error(f"Fetch batch gave up after 5 retries: {last_err}")
    return []


async def main():
    """Continuous queue + N consumers + shared httpx.AsyncClient (Layer A)."""
    global shutdown_requested

    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting Rocket Alumni LLM Contact Extractor")
    logger.info(f"  vLLM URLs ({len(VAST_VLLM_URLS)}): {VAST_VLLM_URLS}")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Batch size: {BATCH_SIZE}, Concurrency: {WORKER_CONCURRENCY}")

    if not await wait_for_vllm_startup():
        logger.error("Failed to connect to vLLM; exiting")
        sys.exit(1)

    work_queue: asyncio.Queue = asyncio.Queue()
    write_buffer: list = []
    write_lock = asyncio.Lock()
    rows_done = 0
    start_time = time.time()
    no_op_semaphore = asyncio.Semaphore(WORKER_CONCURRENCY)

    async def flush_write_buffer():
        nonlocal rows_done
        async with write_lock:
            if write_buffer:
                chunk = list(write_buffer)
                write_buffer.clear()
            else:
                return
        await asyncio.to_thread(upsert_batch_sync, chunk)
        rows_done += len(chunk)
        elapsed = time.time() - start_time
        rate = rows_done / elapsed if elapsed > 0 else 0
        logger.info(f"Progress: {rows_done} rows, rate={rate:.2f}/sec")

    async def consumer(cid: int, http_client: httpx.AsyncClient):
        while True:
            if shutdown_requested:
                break
            try:
                row = await asyncio.wait_for(work_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if row is None:
                work_queue.task_done()
                break
            try:
                result = await process_row(row, no_op_semaphore, http_client)
                if isinstance(result, ContactExtractionResult):
                    async with write_lock:
                        write_buffer.append(result)
                        should_flush = len(write_buffer) >= BATCH_SIZE
                    if should_flush:
                        await flush_write_buffer()
            except Exception as e:
                logger.error(f"Consumer {cid} error: {e}")
            finally:
                work_queue.task_done()

    async def producer():
        empty_count = 0
        while not shutdown_requested:
            # Off-load sync DB call to a thread so event loop stays free for consumers
            batch = await asyncio.to_thread(fetch_batch)
            logger.info(f"Producer: fetched batch of {len(batch)} rows")
            if not batch:
                empty_count += 1
                if empty_count >= 5:
                    logger.info(f"Producer: 5 empty fetches, sending poison pills")
                    for _ in range(WORKER_CONCURRENCY):
                        await work_queue.put(None)
                    return
                await asyncio.sleep(60)
                continue
            empty_count = 0
            for row in batch:
                if shutdown_requested:
                    for _ in range(WORKER_CONCURRENCY):
                        await work_queue.put(None)
                    return
                await work_queue.put(row)

    try:
        async with httpx.AsyncClient() as http_client:
            producer_task = asyncio.create_task(producer())
            consumer_tasks = [
                asyncio.create_task(consumer(i, http_client))
                for i in range(WORKER_CONCURRENCY)
            ]
            await producer_task
            await asyncio.gather(*consumer_tasks, return_exceptions=False)
            await flush_write_buffer()
    except Exception as e:
        logger.error(f"Main worker error: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    rate = rows_done / elapsed if elapsed > 0 else 0
    logger.info(f"DRAIN DONE: {rows_done} rows in {elapsed:.1f}s ({rate:.2f}/sec)")


if __name__ == "__main__":
    asyncio.run(main())
