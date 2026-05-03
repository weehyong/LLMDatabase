#!/usr/bin/env python3
"""
LLMDatabase Interactive REPL

Usage
-----
  python -m llmdb.cli [--data-dir PATH] [--api-key KEY]

Once in the REPL you can type:
  • SQL statements  — executed directly
  • English queries — prefix with '?' or ':ask'
  • .tables         — list tables
  • .describe <tbl> — show schema
  • .status         — memory / buffer / storage report
  • .help           — this help
  • .quit / .exit   — exit
"""

from __future__ import annotations

import argparse
import json
import os
import readline
import sys

from llmdb import Database


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  LLMDatabase — OpenClaw-inspired DBMS for agentic apps       ║
║  Type .help for commands, .quit to exit                      ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Commands
--------
  <SQL>              Execute a SQL statement
  ?<question>        Translate English to SQL and execute
  :ask <question>    Same as ?
  :explain <q>       Show SQL translation without executing
  .tables            List all tables
  .describe <table>  Show table schema
  .status            System status (memory, buffer, WAL)
  .memory            Remaining page-frame memory
  .flush             Flush buffer pool to disk
  .help              Show this help
  .quit / .exit      Exit

Examples
--------
  CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)
  INSERT INTO users (id, name) VALUES (1, 'Alice')
  SELECT * FROM users WHERE id = 1
  ?How many users are there?
  ?Show me the top 3 users
  :explain Who are the users with the highest score?
"""


def _fmt_rows(rows, max_width=120):
    if not rows:
        return "(no rows)"
    cols = list(rows[0].keys())
    col_widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(col_widths[c]) for c in cols)
    sep = "  ".join("-" * col_widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(col_widths[c]) for c in cols))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LLMDatabase interactive REPL")
    parser.add_argument("--data-dir", default="./llmdb_data", help="Database data directory")
    parser.add_argument("--api-key", default=None, help="LLM API key for natural language queries")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("LLMDB_API_KEY")

    print(BANNER)
    print(f"  Data directory : {args.data_dir}")
    if api_key:
        print("  LLM           : enabled (OpenAI-compatible API)")
    else:
        print("  LLM           : rule-based fallback (set LLMDB_API_KEY for full NL support)")
    print()

    db = Database(args.data_dir, llm_api_key=api_key)

    while True:
        try:
            line = input("llmdb> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        # -- Meta commands --
        if line in (".quit", ".exit", "\\q"):
            print("Bye!")
            break
        elif line == ".help":
            print(HELP)
        elif line == ".tables":
            tables = db.tables()
            if tables:
                for t in tables:
                    print(f"  {t}")
            else:
                print("  (no tables)")
        elif line.startswith(".describe "):
            tbl = line.split(None, 1)[1].strip()
            try:
                cols = db.describe(tbl)
                print(f"\n  Table: {tbl}")
                print(f"  {'Column':<20} {'Type':<12} {'Nullable':<10} {'PK'}")
                print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*5}")
                for c in cols:
                    print(f"  {c['name']:<20} {c['type']:<12} {str(c['nullable']):<10} {c['primary_key']}")
            except KeyError as e:
                print(f"  Error: {e}")
        elif line == ".status":
            status = db.status()
            print(json.dumps(status, indent=2))
        elif line == ".memory":
            mem = db.remaining_memory()
            print(f"  Remaining page-frame memory: {mem:,} bytes ({mem // 1024} KiB)")
        elif line == ".flush":
            n = db.flush()
            print(f"  Flushed {n} dirty pages to disk.")

        # -- Natural language: ?<question> or :ask <question> --
        elif line.startswith("?") or line.lower().startswith(":ask "):
            question = line[1:].strip() if line.startswith("?") else line.split(None, 1)[1].strip()
            print(f"  → Translating: {question!r}")
            result = db.ask(question)
            if hasattr(result, "_translation"):
                print(f"  → SQL: {result.sql}")
                print(f"  → Confidence: {result._translation.confidence:.0%}")
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print()
                print(_fmt_rows(result.rows))
                print(f"\n  ({result.row_count} rows, {result.elapsed_ms:.1f}ms)")

        # -- :explain --
        elif line.lower().startswith(":explain "):
            question = line.split(None, 1)[1].strip()
            explanation = db.explain(question)
            print(explanation)

        # -- SQL --
        else:
            result = db.execute(line)
            if result.error:
                print(f"  Error: {result.error}")
            else:
                if result.rows:
                    print()
                    print(_fmt_rows(result.rows))
                    print(f"\n  ({result.row_count} rows, {result.elapsed_ms:.1f}ms)")
                else:
                    print(f"  OK — {result.affected_rows} rows affected ({result.elapsed_ms:.1f}ms)")


if __name__ == "__main__":
    main()
