#!/usr/bin/env python3
"""Explode rocket_llm_contacts → classify into 6 tiers → write 7 NDJSON files.

Reads SUPABASE_RAW_POOLER_URL from env (must be in liquid-lead-engine/.env).
Applies DNC + contacted suppression.
Output: /tmp/rocket_clay_push/tier_<N>.jsonl  (and tier_6_part2.jsonl if needed).

Schema per tier matches the canonical Clay column spec (universal cols + tier-specific).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

OUT_DIR = Path("/tmp/rocket_clay_push")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERIC_PREFIXES = (
    "info","contact","hello","admin","sales","support","office",
    "front","reception","hi","team","school","principal","secretary",
    "enquiry","enquiries","help","ask","mail","email","general","contactus",
    "webmaster","it","tech","main",
)
GP_ARRAY = "ARRAY['" + "','".join(GENERIC_PREFIXES) + "']"
EMAIL_OK = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
EMAIL_BAD = r"(example|sample|yourdomain|youremail|test\.com|domain\.com|email\.com)"

SUMMARY_CACHE_PATH = Path("/tmp/rocket_clay_push/_summaries_cache.json")

QUERY = f"""
WITH dnc_emails AS (
  SELECT DISTINCT lower(email_lower) AS email
  FROM raw_rocket_alumni_dnc WHERE email_lower IS NOT NULL
),
dnc_domains AS (
  SELECT DISTINCT lower(domain) AS domain
  FROM raw_rocket_alumni_dnc WHERE domain IS NOT NULL AND domain ~ '\\.'
),
contacted_emails AS (
  SELECT DISTINCT lower(email_lower) AS email
  FROM raw_rocket_alumni_contacted WHERE email_lower IS NOT NULL
),
src AS (
  SELECT c.domain_key, c.school_name, c.city, c.state,
         NULLIF(trim(c.owner_name),'') AS school_owner_name,
         NULLIF(trim(c.owner_title),'') AS school_owner_title,
         c.contacts, c.generic_emails, c.extracted_at
  FROM rocket_llm_contacts_2026_05_06 c
  WHERE c.extraction_status='success'
    AND lower(c.domain_key) NOT IN (SELECT domain FROM dnc_domains)
),
named AS (
  SELECT s.domain_key AS raw_row_id, s.domain_key, s.school_name, s.city, s.state,
         NULLIF(trim(c->>'name'),'') AS owner_name,
         NULLIF(trim(c->>'title'),'') AS owner_title,
         lower(trim(c->>'email')) AS email,
         NULLIF(trim(c->>'phone'),'') AS phone,
         s.extracted_at
  FROM src s, jsonb_array_elements(s.contacts) c
  WHERE c->>'email' IS NOT NULL
    AND lower(trim(c->>'email')) ~ '{EMAIL_OK}'
    AND lower(trim(c->>'email')) !~ '{EMAIL_BAD}'
),
generic AS (
  SELECT s.domain_key AS raw_row_id, s.domain_key, s.school_name, s.city, s.state,
         s.school_owner_name AS owner_name, s.school_owner_title AS owner_title,
         lower(trim(g, '"')) AS email, NULL::text AS phone, s.extracted_at
  FROM src s, jsonb_array_elements_text(s.generic_emails) g
  WHERE lower(trim(g, '"')) ~ '{EMAIL_OK}'
    AND lower(trim(g, '"')) !~ '{EMAIL_BAD}'
),
no_contact AS (
  SELECT s.domain_key AS raw_row_id, s.domain_key, s.school_name, s.city, s.state,
         s.school_owner_name AS owner_name, s.school_owner_title AS owner_title,
         NULL::text AS email, NULL::text AS phone, s.extracted_at
  FROM src s
),
suppress AS (
  SELECT email FROM dnc_emails UNION SELECT email FROM contacted_emails
),
unioned AS (
  SELECT * FROM named WHERE email NOT IN (SELECT email FROM suppress)
  UNION
  SELECT * FROM generic WHERE email NOT IN (SELECT email FROM suppress)
  UNION ALL
  SELECT * FROM no_contact
  WHERE no_contact.domain_key NOT IN (
    SELECT domain_key FROM named WHERE email NOT IN (SELECT email FROM suppress)
    UNION
    SELECT domain_key FROM generic WHERE email NOT IN (SELECT email FROM suppress)
  )
),
classified AS (
  SELECT *,
    CASE WHEN email IS NOT NULL THEN
      CASE WHEN split_part(email,'@',1) = ANY({GP_ARRAY}) THEN 'generic' ELSE 'personal' END
    ELSE NULL END AS email_kind
  FROM unioned
),
tiered AS (
  SELECT *,
    CASE
      WHEN owner_name IS NOT NULL AND email_kind = 'personal' THEN 1
      WHEN owner_name IS NOT NULL AND email_kind = 'generic'  THEN 2
      WHEN owner_name IS NULL     AND email_kind = 'personal' THEN 3
      WHEN owner_name IS NULL     AND email_kind = 'generic'  THEN 4
      WHEN owner_name IS NOT NULL AND email IS NULL           THEN 5
      ELSE 6
    END AS tier
  FROM classified
)
SELECT tier, raw_row_id, domain_key, school_name, city, state,
       owner_name, owner_title, email, email_kind, phone, extracted_at
FROM tiered
ORDER BY tier, domain_key, email NULLS LAST;
"""


def shape_row(row: dict, summaries: dict) -> dict:
    """Normalise one row into Clay-ready dict (string-typed where Clay needs it)."""
    domain = row["domain_key"]
    summary = summaries.get(domain, "")
    base = {
        "school_name": row["school_name"] or "",
        "domain": domain,
        "website": f"https://{domain}",
        "city": row["city"] or "",
        "state": row["state"] or "",
        "place_id": domain,
        "raw_row_id": row["raw_row_id"],
        "tier": int(row["tier"]),
        "source": "gmaps+llm_extract_v1",
        "icp_score": None,
        "icp_reason": "",
        "extracted_at": row["extracted_at"].isoformat() if row["extracted_at"] else "",
        "website_summary_8k": (summary or "")[:8000],
    }
    t = int(row["tier"])
    if t in (1, 2, 3, 4):
        base["email"] = row["email"] or ""
        base["email_kind"] = row["email_kind"] or ""
        base["phone"] = row["phone"] or ""
        if t in (1, 2):
            base["owner_name"] = row["owner_name"] or ""
            base["owner_title"] = row["owner_title"] or ""
    elif t == 5:
        base["owner_name"] = row["owner_name"] or ""
        base["owner_title"] = row["owner_title"] or ""
        base["phone"] = row["phone"] or ""
    elif t == 6:
        base["phone"] = row["phone"] or ""
    return base


def main() -> int:
    dsn_raw = os.environ.get("SUPABASE_RAW_POOLER_URL")
    if not dsn_raw:
        print("ERROR: SUPABASE_RAW_POOLER_URL not set", file=sys.stderr)
        return 2
    dsn = dsn_raw.replace(":5432", ":6543")

    if SUMMARY_CACHE_PATH.exists():
        print(f"Loading summary cache from {SUMMARY_CACHE_PATH} ...", flush=True)
        summaries = json.loads(SUMMARY_CACHE_PATH.read_text())
        print(f"  loaded {len(summaries):,} summaries (avg {sum(len(v) for v in summaries.values())//max(len(summaries),1):,} chars)", flush=True)
    else:
        print("WARN: no summary cache; rows will have empty website_summary_8k", file=sys.stderr)
        summaries = {}

    print("Connecting to Supabase TX pooler …", flush=True)
    conn = psycopg2.connect(dsn, connect_timeout=30)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SET statement_timeout = '120s'")
    conn.commit()
    conn.set_session(readonly=True)
    cur.itersize = 5000

    files = {t: open(OUT_DIR / f"tier_{t}.jsonl", "w") for t in (1, 2, 3, 4, 5, 6)}
    counts = {t: 0 for t in files}
    print("Streaming classified rows …", flush=True)
    cur.execute(QUERY)
    for row in cur:
        t = int(row["tier"])
        out = shape_row(dict(row), summaries)
        files[t].write(json.dumps(out, ensure_ascii=False) + "\n")
        counts[t] += 1
    for f in files.values():
        f.close()
    conn.close()

    # Split T6 if > 50K
    t6_path = OUT_DIR / "tier_6.jsonl"
    if counts[6] > 50000:
        with t6_path.open() as fh:
            lines = fh.readlines()
        with t6_path.open("w") as fh:
            fh.writelines(lines[:50000])
        with (OUT_DIR / "tier_6_part2.jsonl").open("w") as fh:
            fh.writelines(lines[50000:])
        counts["6_part2"] = len(lines) - 50000
        counts[6] = 50000

    print("\n=== Tier file counts ===")
    for t in (1, 2, 3, 4, 5, 6):
        print(f"  tier_{t}.jsonl     : {counts[t]:>7,}")
    if "6_part2" in counts:
        print(f"  tier_6_part2.jsonl : {counts['6_part2']:>7,}")
    total = sum(counts.values())
    print(f"  TOTAL              : {total:>7,}")
    print(f"\nWrote to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
