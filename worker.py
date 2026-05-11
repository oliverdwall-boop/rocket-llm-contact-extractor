#!/usr/bin/env python3
"""
Bruin Corp LLM Contact Extractor
Pulls website summaries from Supabase (v_bruin_llm_input_2026_05_08),
calls Vast.ai vLLM (Qwen 2.5-7B), extracts structured contact JSON,
upserts one row per contact to bruin_llm_contacts_staging_2026_05_08.

Forked from Rocket Alumni extractor — adapted for construction-business schema:
  input:  domain, company_name, city, state, website_text
  output: domain, contact_name, contact_title, contact_email, contact_phone,
          contact_linkedin, llm_raw_response, llm_status
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

import httpx
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_RAW_POOLER_URL", "postgres://...")
_RAW_URLS = os.getenv("VAST_VLLM_URL", "PLACEHOLDER")
VAST_VLLM_URLS = [u.strip().rstrip("/") for u in _RAW_URLS.split(",") if u.strip()]
import itertools as _itertools, threading as _threading
_url_cycle = _itertools.cycle(VAST_VLLM_URLS) if VAST_VLLM_URLS else _itertools.cycle(["PLACEHOLDER"])
_url_lock = _threading.Lock()
def _next_vllm_url() -> str:
    with _url_lock:
        return next(_url_cycle)

LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "200"))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
INPUT_VIEW = os.getenv("INPUT_VIEW", "public.v_bruin_llm_input_2026_05_08")
OUTPUT_TABLE = os.getenv("OUTPUT_TABLE", "public.bruin_llm_contacts_staging_2026_05_08")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "200"))
TEMPERATURE_INITIAL = float(os.getenv("TEMPERATURE", "0.1"))
HEALTHCHECK_RETRIES = int(os.getenv("HEALTHCHECK_RETRIES", "3"))
HEALTHCHECK_DELAY_SEC = int(os.getenv("HEALTHCHECK_DELAY_SEC", "5"))

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
    """One row per business domain — contacts array exploded on write."""
    domain: str
    contacts: list          # [{name, title, email, phone}]
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
    return psycopg2.connect(SUPABASE_URL)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def healthcheck_vllm():
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
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_prompt(row: dict) -> tuple[str, str]:
    """Build system and user prompt for bruin construction-business contact extraction."""
    company_name = row.get("company_name") or row.get("name") or ""
    city = row.get("city") or ""
    state = row.get("state") or ""
    website_text = row.get("website_text") or ""

    system = (
        "You extract contact information from US construction-industry "
        "business websites (general contractors, commercial real estate, "
        "home inspectors, builders, renovators). "
        "Output STRICT JSON only — no markdown, no explanations, no code fences."
    )
    user = f"""Business: {company_name} ({city}, {state})

Website content:
{website_text[:6000]}

Extract every named person mentioned (owners, founders, principals, CEOs, presidents, project managers, estimators, key staff). For each person, capture their title, email, and phone if present in the text.

Return ONLY this JSON structure:
{{
  "contacts": [
    {{"name": "Jane Smith", "title": "President", "email": "jane@example.com", "phone": "+1-555-0100"}}
  ],
  "generic_emails": ["info@example.com", "ops@example.com"],
  "owner_name": "Jane Smith",
  "owner_title": "President"
}}

Rules:
- Only extract data EXPLICITLY present in the text. Never fabricate.
- contacts: array of named people. Empty array if none found.
- generic_emails: role-based emails (info@, contact@, sales@). Empty array if none.
- owner_name/owner_title: single top executive if clearly named. Empty string if ambiguous.
- Return valid JSON. No prose. No code fences."""
    return system, user


async def call_vllm(
    system: str, user: str, temperature: float = TEMPERATURE_INITIAL,
    http_client: Optional[httpx.AsyncClient] = None,
) -> Optional[str]:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": temperature,
    }
    try:
        payload["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    headers = {}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    base = _next_vllm_url().rstrip("/")
    completions_path = "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"

    for attempt in range(5):
        try:
            # Reuse shared client if provided (avoids per-request TCP handshake)
            if http_client is not None:
                _client_ctx = None
                client = http_client
            else:
                _client_ctx = httpx.AsyncClient(timeout=60.0)
                client = await _client_ctx.__aenter__()
            try:
                resp = await client.post(
                    f"{base}{completions_path}",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
            finally:
                if _client_ctx is not None:
                    await _client_ctx.__aexit__(None, None, None)
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                try:
                    wait_s = float(retry_after) if retry_after else 5 * (2 ** attempt)
                except ValueError:
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
    async with semaphore:
        # Use domain as the key (bruin view exposes domain or root_domain)
        domain = (
            row.get("domain")
            or row.get("root_domain")
            or row.get("website")
            or str(row.get("id", ""))
        )

        start = time.time()
        system, user = build_prompt(row)
        retry_count = 0
        temperature = TEMPERATURE_INITIAL
        parsed = None
        raw_response = ""

        for attempt in range(3):
            raw_response = await call_vllm(system, user, temperature, http_client=client_http)
            if raw_response:
                parsed = parse_response(raw_response)
                if parsed:
                    retry_count = attempt
                    break
            temperature = [0.1, 0.3, 0.5][attempt]

        latency_ms = (time.time() - start) * 1000

        if parsed:
            extraction_status = "ok"
            contacts = parsed.get("contacts", []) or []
            generic_emails = parsed.get("generic_emails", []) or []
            owner_name = parsed.get("owner_name", "") or ""
            owner_title = parsed.get("owner_title", "") or ""
        else:
            extraction_status = "parse_failed"
            contacts = []
            generic_emails = []
            owner_name = ""
            owner_title = ""

        return ContactExtractionResult(
            domain=domain,
            contacts=contacts,
            generic_emails=generic_emails,
            owner_name=owner_name,
            owner_title=owner_title,
            model_used=MODEL_NAME,
            extracted_at=datetime.utcnow().isoformat(),
            latency_ms=latency_ms,
            raw_response=raw_response[:4000] if raw_response else "",
            retry_count=retry_count,
            extraction_status=extraction_status,
            source_summary_chars=len(row.get("website_text", "") or ""),
        )


def upsert_batch_sync(results: list):
    """
    Explode contacts array into one staging row per contact.
    Schema: domain, contact_name, contact_title, contact_email, contact_phone,
            contact_linkedin, llm_raw_response, llm_status, extracted_at
    """
    if not results:
        return
    rows_to_insert = []
    for r in results:
        # Always write at least one row per domain (even if no contacts found)
        if r.contacts:
            for c in r.contacts:
                if not isinstance(c, dict):
                    continue
                rows_to_insert.append((
                    r.domain,
                    c.get("name") or None,
                    c.get("title") or None,
                    c.get("email") or None,
                    c.get("phone") or None,
                    c.get("linkedin") or None,
                    r.raw_response[:4000] if r.raw_response else None,
                    r.extraction_status,
                ))
        else:
            # No contacts found — write a single sentinel row so we don't re-process
            rows_to_insert.append((
                r.domain,
                r.owner_name or None,
                r.owner_title or None,
                None,
                None,
                None,
                r.raw_response[:4000] if r.raw_response else None,
                r.extraction_status,
            ))

    if not rows_to_insert:
        return

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        extras.execute_values(
            cur,
            f"""
            INSERT INTO {OUTPUT_TABLE}
            (domain, contact_name, contact_title, contact_email, contact_phone,
             contact_linkedin, llm_raw_response, llm_status)
            VALUES %s
            """,
            rows_to_insert,
            page_size=200,
        )
        conn.commit()
        logger.info(f"Upserted {len(rows_to_insert)} contact rows from {len(results)} domains")
    except Exception as e:
        logger.error(f"Upsert failed: {e}")
        conn.rollback()
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()


def fetch_batch() -> list[dict]:
    """Fetch next batch from input view. Excludes already-processed domains."""
    last_err = None
    for attempt in range(5):
        conn = None
        cur = None
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=extras.RealDictCursor)
            # Exclude domains already in staging so we don't reprocess
            cur.execute(
                f"""
                SELECT v.*
                FROM {INPUT_VIEW} v
                WHERE v.domain NOT IN (
                    SELECT domain FROM {OUTPUT_TABLE}
                )
                ORDER BY v.domain
                LIMIT %s
                """,
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
            time.sleep(2 ** attempt)
    logger.error(f"Fetch batch gave up after 5 retries: {last_err}")
    return []


async def main():
    global shutdown_requested

    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting Bruin Corp LLM Contact Extractor")
    logger.info(f"  Input view:  {INPUT_VIEW}")
    logger.info(f"  Output table: {OUTPUT_TABLE}")
    logger.info(f"  vLLM URLs ({len(VAST_VLLM_URLS)}): {VAST_VLLM_URLS}")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Batch size: {BATCH_SIZE}, Concurrency: {WORKER_CONCURRENCY}")
    logger.info(f"  Max output tokens: {MAX_OUTPUT_TOKENS}")

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
        logger.info(f"Progress: {rows_done} domains processed, rate={rate:.2f}/sec")

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
            batch = await asyncio.to_thread(fetch_batch)
            logger.info(f"Producer: fetched batch of {len(batch)} rows")
            if not batch:
                empty_count += 1
                if empty_count >= 5:
                    logger.info(f"Producer: 5 empty fetches, all domains processed — sending poison pills")
                    for _ in range(WORKER_CONCURRENCY):
                        await work_queue.put(None)
                    return
                await asyncio.sleep(30)
                continue
            empty_count = 0
            for row in batch:
                if shutdown_requested:
                    for _ in range(WORKER_CONCURRENCY):
                        await work_queue.put(None)
                    return
                await work_queue.put(row)

    try:
        # Raise connection pool limits so 200 concurrent coroutines don't queue on socket acquisition
        _limits = httpx.Limits(max_connections=WORKER_CONCURRENCY + 20, max_keepalive_connections=WORKER_CONCURRENCY)
        async with httpx.AsyncClient(timeout=60.0, limits=_limits) as http_client:
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
    logger.info(f"DRAIN DONE: {rows_done} domains in {elapsed:.1f}s ({rate:.2f}/sec)")


if __name__ == "__main__":
    asyncio.run(main())
