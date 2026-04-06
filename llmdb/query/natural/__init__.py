"""
Natural Language → SQL Translator — the LLM-powered evolution of SQL.

The Problem
-----------
Traditional SQL is a powerful but unnatural language.  Users must know exact
table/column names, understand JOIN semantics, and remember the precise
syntax for aggregations, window functions, and nested sub-queries.

The Evolution
-------------
LLMDatabase replaces the SQL surface layer with plain English.  Instead of:

    SELECT customers.name, SUM(orders.amount)
    FROM customers
    JOIN orders ON customers.id = orders.customer_id
    WHERE orders.created_at > '2024-01-01'
    GROUP BY customers.name
    ORDER BY SUM(orders.amount) DESC
    LIMIT 10;

the user writes:

    "Who are the top 10 customers by total order value since January 2024?"

The translator uses an LLM (or, when no LLM key is configured, a
rule-based fallback) to:
  1. Parse user intent from the English sentence.
  2. Consult the schema catalog to resolve table/column names.
  3. Emit a SQL string that the downstream parser can handle.
  4. Optionally explain its reasoning (chain-of-thought).

Design
------
The translator is pluggable:
  - NaturalLanguageTranslator  (abstract base)
  - LLMTranslator              (calls any OpenAI-compatible LLM API)
  - RuleBasedTranslator        (keyword-matching fallback, zero dependencies)

The rule-based fallback handles the most common patterns without needing
an LLM key, making the system useful out of the box.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# TranslationResult
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """Returned by every translator."""
    sql: str
    confidence: float          # 0.0–1.0
    explanation: str           # Human-readable reasoning
    alternatives: List[str] = field(default_factory=list)
    used_tables: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"SQL        : {self.sql}",
            f"Confidence : {self.confidence:.0%}",
            f"Reasoning  : {self.explanation}",
        ]
        if self.alternatives:
            lines.append("Alternatives:")
            for alt in self.alternatives:
                lines.append(f"  • {alt}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class NaturalLanguageTranslator(ABC):
    """Translate an English sentence into a SQL string given a schema context."""

    @abstractmethod
    def translate(self, text: str, schema_context: Dict[str, Any]) -> TranslationResult: ...

    def batch_translate(
        self, texts: List[str], schema_context: Dict[str, Any]
    ) -> List[TranslationResult]:
        return [self.translate(t, schema_context) for t in texts]


# ---------------------------------------------------------------------------
# Schema Context builder (used by both translators)
# ---------------------------------------------------------------------------

def build_schema_context(catalog: Any) -> Dict[str, Any]:
    """Summarise the catalog into a compact dict for prompt injection."""
    tables = {}
    for name in catalog.list_tables():
        obj = catalog.get_table(name)
        if obj is None:
            continue
        from llmdb.core.actors import ColumnSetComponent
        col_comp = obj.get_component_typed(ColumnSetComponent)
        columns = []
        if col_comp:
            for col in col_comp.columns:
                columns.append({
                    "name": col.name,
                    "type": col.col_type.name,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                })
        tables[name] = {"columns": columns}
    return {"tables": tables}


# ---------------------------------------------------------------------------
# Rule-Based Translator — zero-dependency fallback
# ---------------------------------------------------------------------------

class RuleBasedTranslator(NaturalLanguageTranslator):
    """Pattern-match English to SQL using hand-crafted rules.

    Covers the most common intents without any LLM:
      - "show me all X"  → SELECT * FROM x
      - "how many X"     → SELECT COUNT(*) FROM x
      - "find X where Y" → SELECT * FROM x WHERE y
      - "insert X into Y" → INSERT INTO y ...
      - "delete X from Y" → DELETE FROM y WHERE ...
      - "what is the total/sum/average of X"  → aggregate SELECT
    """

    # Intent patterns: (regex, handler_method_name)
    _INTENT_PATTERNS = [
        (r"how many (.+?)(?:\s+are there)?(?:\s+in\s+(.+))?$", "_handle_count"),
        (r"(?:show|list|get|find|select|give me|display|fetch) (?:me )?(?:all |the )?(.+?)(?:\s+(?:where|in which|that)\s+(.+))?$", "_handle_select"),
        (r"(?:what is|what are|calculate|compute) (?:the )?(?:total|sum) of (.+?) (?:in|from|of) (.+)$", "_handle_sum"),
        (r"(?:what is|what are|calculate|compute) (?:the )?(?:average|avg|mean) (?:of )?(.+?) (?:in|from|of) (.+)$", "_handle_avg"),
        (r"(?:add|insert|create) (?:a |an |new )?(.+?) (?:with|having|where) (.+)$", "_handle_insert_hint"),
        (r"(?:delete|remove|drop) (.+?) (?:where|with|that has) (.+)$", "_handle_delete"),
        (r"(?:top|highest|largest|biggest) (\d+) (.+?) (?:by|ordered by|sorted by) (.+)$", "_handle_top_n"),
        (r"(.+?) (?:that (?:have|has)|with|having) (.+)$", "_handle_filter"),
    ]

    def translate(self, text: str, schema_context: Dict[str, Any]) -> TranslationResult:
        cleaned = text.strip().rstrip("?!.").lower()
        tables = list(schema_context.get("tables", {}).keys())

        # Try each intent pattern
        for pattern, handler_name in self._INTENT_PATTERNS:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                handler = getattr(self, handler_name)
                result = handler(m, cleaned, tables, schema_context)
                if result:
                    return result

        # Fallback: guess the table from any word matching a known table name
        matched_table = self._guess_table(cleaned, tables)
        sql = f"SELECT * FROM {matched_table}" if matched_table else "SELECT 1"
        return TranslationResult(
            sql=sql,
            confidence=0.3,
            explanation=(
                f"Could not parse a specific intent. "
                f"Falling back to a full scan of '{matched_table}'."
                if matched_table
                else "Could not determine intent or table."
            ),
            used_tables=[matched_table] if matched_table else [],
        )

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_count(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        subject = m.group(1).strip()
        location = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else None
        table = location or self._guess_table(subject, tables) or subject
        table = self._canonical_table(table, tables)
        return TranslationResult(
            sql=f"SELECT COUNT(*) AS count FROM {table}",
            confidence=0.85,
            explanation=f"Counting all rows in '{table}'.",
            used_tables=[table],
        )

    def _handle_select(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        subject = m.group(1).strip()
        condition = m.group(2).strip() if m.lastindex >= 2 and m.group(2) else None
        table = self._guess_table(subject, tables) or self._canonical_table(subject, tables)
        if not table:
            return None
        cols = self._guess_columns(subject, table, ctx)
        col_str = ", ".join(cols) if cols else "*"
        where_clause = ""
        if condition:
            where_clause = f" WHERE {self._natural_condition_to_sql(condition, table, ctx)}"
        return TranslationResult(
            sql=f"SELECT {col_str} FROM {table}{where_clause}",
            confidence=0.75,
            explanation=f"Selecting {col_str} from '{table}'" + (f" with filter: {condition}" if condition else "."),
            used_tables=[table],
        )

    def _handle_sum(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        col_hint = m.group(1).strip()
        table_hint = m.group(2).strip()
        table = self._canonical_table(table_hint, tables)
        col = self._guess_column(col_hint, table, ctx) or col_hint
        return TranslationResult(
            sql=f"SELECT SUM({col}) AS total FROM {table}",
            confidence=0.80,
            explanation=f"Summing '{col}' in '{table}'.",
            used_tables=[table],
        )

    def _handle_avg(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        col_hint = m.group(1).strip()
        table_hint = m.group(2).strip()
        table = self._canonical_table(table_hint, tables)
        col = self._guess_column(col_hint, table, ctx) or col_hint
        return TranslationResult(
            sql=f"SELECT AVG({col}) AS average FROM {table}",
            confidence=0.80,
            explanation=f"Averaging '{col}' in '{table}'.",
            used_tables=[table],
        )

    def _handle_insert_hint(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        subject = m.group(1).strip()
        attrs = m.group(2).strip()
        table = self._canonical_table(subject, tables)
        return TranslationResult(
            sql=f"-- INSERT INTO {table} (...) VALUES (...);  -- fill in values for: {attrs}",
            confidence=0.5,
            explanation=f"An INSERT into '{table}' with attributes: {attrs}. Please fill in the exact values.",
            used_tables=[table],
        )

    def _handle_delete(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        subject = m.group(1).strip()
        condition = m.group(2).strip()
        table = self._canonical_table(subject, tables)
        where = self._natural_condition_to_sql(condition, table, ctx)
        return TranslationResult(
            sql=f"DELETE FROM {table} WHERE {where}",
            confidence=0.70,
            explanation=f"Deleting from '{table}' where {condition}.",
            used_tables=[table],
        )

    def _handle_top_n(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        n = int(m.group(1))
        subject = m.group(2).strip()
        order_col_hint = m.group(3).strip()
        table = self._canonical_table(subject, tables) or self._guess_table(subject, tables)
        if not table:
            return None
        col = self._guess_column(order_col_hint, table, ctx) or order_col_hint
        return TranslationResult(
            sql=f"SELECT * FROM {table} ORDER BY {col} DESC LIMIT {n}",
            confidence=0.80,
            explanation=f"Top {n} rows from '{table}' ordered by '{col}' descending.",
            used_tables=[table],
        )

    def _handle_filter(self, m, text, tables, ctx) -> Optional[TranslationResult]:
        subject = m.group(1).strip()
        condition = m.group(2).strip()
        table = self._guess_table(subject, tables)
        if not table:
            return None
        where = self._natural_condition_to_sql(condition, table, ctx)
        return TranslationResult(
            sql=f"SELECT * FROM {table} WHERE {where}",
            confidence=0.65,
            explanation=f"Filtering '{table}' where {condition}.",
            used_tables=[table],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _guess_table(self, text: str, tables: List[str]) -> Optional[str]:
        text_lower = text.lower()
        # Exact match first
        for t in tables:
            if t.lower() in text_lower:
                return t
        # Singular/plural heuristic
        for t in tables:
            singular = t.rstrip("s").lower()
            if singular and singular in text_lower:
                return t
        return None

    def _canonical_table(self, name: str, tables: List[str]) -> str:
        for t in tables:
            if t.lower() == name.lower():
                return t
            if t.lower() + "s" == name.lower() or t.lower() == name.lower() + "s":
                return t
        return name

    def _guess_columns(self, text: str, table: str, ctx: Dict) -> List[str]:
        cols = ctx.get("tables", {}).get(table, {}).get("columns", [])
        text_lower = text.lower()
        matched = [c["name"] for c in cols if c["name"].lower() in text_lower]
        return matched if matched else []

    def _guess_column(self, hint: str, table: str, ctx: Dict) -> Optional[str]:
        cols = ctx.get("tables", {}).get(table, {}).get("columns", [])
        hint_lower = hint.lower()
        for col in cols:
            if col["name"].lower() in hint_lower or hint_lower in col["name"].lower():
                return col["name"]
        return None

    def _natural_condition_to_sql(self, text: str, table: str, ctx: Dict) -> str:
        """Crude English condition → SQL predicate."""
        result = text
        # "is / equals / equal to X" → "= X"
        result = re.sub(r"\b(?:is|equals?|equal to)\s+(\S+)", r"= '\1'", result)
        result = re.sub(r"\b(?:is|equals?|equal to)\s+(\d+(?:\.\d+)?)", r"= \1", result)
        # "greater than / more than X" → "> X"
        result = re.sub(r"\b(?:greater than|more than|above|over)\s+(\d+(?:\.\d+)?)", r"> \1", result)
        # "less than / below X" → "< X"
        result = re.sub(r"\b(?:less than|below|under|fewer than)\s+(\d+(?:\.\d+)?)", r"< \1", result)
        # "between X and Y" stays
        # "contains X" → "LIKE '%X%'"
        result = re.sub(r"\bcontains?\s+(\S+)", r"LIKE '%\1%'", result)
        # "starts with X" → "LIKE 'X%'"
        result = re.sub(r"\bstarts? with\s+(\S+)", r"LIKE '\1%'", result)
        # "ends with X" → "LIKE '%X'"
        result = re.sub(r"\bends? with\s+(\S+)", r"LIKE '%\1'", result)
        # "not X" → "!= X" or "IS NOT NULL"
        result = re.sub(r"\bnot null\b", "IS NOT NULL", result, flags=re.IGNORECASE)
        result = re.sub(r"\bnot\s+(\S+)", r"!= '\1'", result)
        return result


# ---------------------------------------------------------------------------
# LLM-Powered Translator
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert database administrator and SQL translator.
Your job is to convert natural English questions into precise SQL queries.

Rules:
1. Use ONLY the tables and columns described in the schema context.
2. Prefer simple queries over complex ones when both would work.
3. Always include a brief explanation of your reasoning.
4. Return your response as a JSON object with keys:
   - "sql": the SQL query string
   - "confidence": a float 0.0–1.0
   - "explanation": one or two sentences describing your reasoning
   - "alternatives": a list of 0–2 alternative SQL strings (optional)
   - "used_tables": list of table names referenced

Schema context:
{schema}

Only respond with valid JSON. Do not include markdown code fences.
"""


class LLMTranslator(NaturalLanguageTranslator):
    """Calls an OpenAI-compatible chat-completion API to translate English → SQL.

    Parameters
    ----------
    api_key    : OpenAI (or compatible) API key.
    model      : Model name (default: gpt-4o-mini).
    base_url   : Base URL for the API endpoint.
    fallback   : If the LLM call fails, fall back to RuleBasedTranslator.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        fallback: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._fallback = RuleBasedTranslator() if fallback else None
        self._timeout = timeout

    def translate(self, text: str, schema_context: Dict[str, Any]) -> TranslationResult:
        schema_str = json.dumps(schema_context, indent=2)
        system = _SYSTEM_PROMPT.format(schema=schema_str)
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "temperature": 0.1,
            "max_tokens": 512,
        }).encode()

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())
            raw = body["choices"][0]["message"]["content"]
            data = json.loads(raw)
            return TranslationResult(
                sql=data.get("sql", ""),
                confidence=float(data.get("confidence", 0.9)),
                explanation=data.get("explanation", ""),
                alternatives=data.get("alternatives", []),
                used_tables=data.get("used_tables", []),
            )
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            if self._fallback:
                result = self._fallback.translate(text, schema_context)
                result.explanation = (
                    f"[LLM unavailable: {exc}] Fell back to rule-based translation. "
                    + result.explanation
                )
                result.confidence *= 0.6
                return result
            raise


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_translator(api_key: Optional[str] = None, **kwargs: Any) -> NaturalLanguageTranslator:
    """Return the best available translator given the environment."""
    if api_key:
        return LLMTranslator(api_key=api_key, **kwargs)
    return RuleBasedTranslator()
