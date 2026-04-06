"""
SQL Parser — converts a SQL string into an Abstract Syntax Tree (AST).

This is a hand-written recursive-descent parser that covers a practical
subset of SQL:

  DDL  : CREATE TABLE, DROP TABLE, CREATE INDEX, DROP INDEX
  DML  : SELECT (with WHERE, ORDER BY, LIMIT, JOINs), INSERT, UPDATE, DELETE
  TCL  : BEGIN, COMMIT, ROLLBACK

The parser produces dataclass AST nodes that the planner and executor consume.
Adding new SQL constructs means adding a new AST node + a parse_* method.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------

class TType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    SEMICOLON = auto()
    DOT = auto()
    STAR = auto()
    EOF = auto()


KEYWORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
    "DELETE", "CREATE", "TABLE", "DROP", "INDEX", "ON", "UNIQUE", "PRIMARY",
    "KEY", "FOREIGN", "REFERENCES", "NOT", "NULL", "DEFAULT", "CHECK",
    "BEGIN", "COMMIT", "ROLLBACK", "TRANSACTION", "ORDER", "BY", "ASC",
    "DESC", "LIMIT", "OFFSET", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER",
    "FULL", "CROSS", "NATURAL", "AND", "OR", "IN", "IS", "LIKE", "BETWEEN",
    "EXISTS", "ALL", "ANY", "SOME", "DISTINCT", "GROUP", "HAVING",
    "INTEGER", "INT", "REAL", "FLOAT", "TEXT", "BLOB", "BOOLEAN", "BOOL",
    "TIMESTAMP", "DATETIME", "JSON", "VARCHAR", "CHAR",
    "IF", "EXISTS", "CASCADE", "RESTRICT", "NO", "ACTION",
    "ALTER", "ADD", "COLUMN", "RENAME", "TO", "VIEW", "AS",
    "RETURNING", "WITH",
}


@dataclass
class Token:
    ttype: TType
    value: str
    pos: int

    def __repr__(self) -> str:
        return f"Token({self.ttype.name}, {self.value!r})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"(?P<STRING>'(?:''|[^'])*')"
    r"|(?P<NUMBER>-?\d+(?:\.\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP>[<>!=]+|[+\-*/])"
    r"|(?P<LPAREN>\()"
    r"|(?P<RPAREN>\))"
    r"|(?P<COMMA>,)"
    r"|(?P<SEMI>;)"
    r"|(?P<DOT>\.)"
    r"|(?P<STAR>\*)"
    r"|(?P<WS>\s+)",
    re.IGNORECASE,
)


def tokenize(sql: str) -> List[Token]:
    tokens: List[Token] = []
    for m in _TOKEN_RE.finditer(sql):
        kind = m.lastgroup
        value = m.group()
        pos = m.start()
        if kind == "WS":
            continue
        elif kind == "STRING":
            tokens.append(Token(TType.STRING, value[1:-1].replace("''", "'"), pos))
        elif kind == "NUMBER":
            tokens.append(Token(TType.NUMBER, value, pos))
        elif kind == "IDENT":
            upper = value.upper()
            if upper in KEYWORDS:
                tokens.append(Token(TType.KEYWORD, upper, pos))
            else:
                tokens.append(Token(TType.IDENTIFIER, value, pos))
        elif kind == "OP":
            tokens.append(Token(TType.OPERATOR, value, pos))
        elif kind == "LPAREN":
            tokens.append(Token(TType.LPAREN, "(", pos))
        elif kind == "RPAREN":
            tokens.append(Token(TType.RPAREN, ")", pos))
        elif kind == "COMMA":
            tokens.append(Token(TType.COMMA, ",", pos))
        elif kind == "SEMI":
            tokens.append(Token(TType.SEMICOLON, ";", pos))
        elif kind == "DOT":
            tokens.append(Token(TType.DOT, ".", pos))
        elif kind == "STAR":
            tokens.append(Token(TType.STAR, "*", pos))
    tokens.append(Token(TType.EOF, "", len(sql)))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    name: str
    col_type: str
    nullable: bool = True
    primary_key: bool = False
    default: Optional[str] = None
    unique: bool = False


@dataclass
class CreateTableStmt:
    table_name: str
    columns: List[ColumnSpec]
    if_not_exists: bool = False


@dataclass
class DropTableStmt:
    table_name: str
    if_exists: bool = False


@dataclass
class CreateIndexStmt:
    index_name: str
    table_name: str
    columns: List[str]
    unique: bool = False


@dataclass
class DropIndexStmt:
    index_name: str


@dataclass
class Expr:
    """A scalar expression node — simplified to a string for now."""
    text: str        # raw SQL text of the expression
    kind: str = "raw"  # raw | col_ref | literal | call | binop


@dataclass
class JoinClause:
    join_type: str   # INNER | LEFT | RIGHT | FULL | CROSS
    table_name: str
    alias: Optional[str]
    condition: Optional[Expr]


@dataclass
class SelectStmt:
    columns: List[str]           # list of "expr [AS alias]" strings, or ["*"]
    from_table: Optional[str]
    table_alias: Optional[str]
    joins: List[JoinClause]
    where: Optional[Expr]
    order_by: List[Tuple[str, str]]   # [(col, ASC|DESC)]
    limit: Optional[int]
    offset: Optional[int]
    distinct: bool = False
    group_by: List[str] = field(default_factory=list)
    having: Optional[Expr] = None


@dataclass
class InsertStmt:
    table_name: str
    columns: List[str]
    values: List[List[Any]]


@dataclass
class UpdateStmt:
    table_name: str
    assignments: Dict[str, Any]   # col → value
    where: Optional[Expr]


@dataclass
class DeleteStmt:
    table_name: str
    where: Optional[Expr]


@dataclass
class BeginStmt:
    pass


@dataclass
class CommitStmt:
    pass


@dataclass
class RollbackStmt:
    pass


# Type alias for any statement node
Stmt = (
    CreateTableStmt | DropTableStmt | CreateIndexStmt | DropIndexStmt
    | SelectStmt | InsertStmt | UpdateStmt | DeleteStmt
    | BeginStmt | CommitStmt | RollbackStmt
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


class Parser:
    """Recursive-descent SQL parser.

    The structure mirrors OpenClaw's XML blueprint parsing: each grammar rule
    is a method; the parser maintains a cursor (_pos) into the token stream.
    """

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, ttype: TType, value: Optional[str] = None) -> Token:
        tok = self._advance()
        if tok.ttype != ttype:
            raise ParseError(
                f"Expected {ttype.name}"
                + (f" '{value}'" if value else "")
                + f" but got {tok}"
            )
        if value and tok.value.upper() != value.upper():
            raise ParseError(f"Expected keyword '{value}' but got '{tok.value}'")
        return tok

    def _match_keyword(self, *kws: str) -> bool:
        tok = self._peek()
        return tok.ttype == TType.KEYWORD and tok.value.upper() in {k.upper() for k in kws}

    def _consume_keyword(self, *kws: str) -> Optional[Token]:
        if self._match_keyword(*kws):
            return self._advance()
        return None

    def _expect_keyword(self, *kws: str) -> Token:
        tok = self._advance()
        if tok.ttype != TType.KEYWORD or tok.value.upper() not in {k.upper() for k in kws}:
            raise ParseError(f"Expected keyword {kws} but got {tok}")
        return tok

    def _match(self, ttype: TType, value: Optional[str] = None) -> bool:
        tok = self._peek()
        if tok.ttype != ttype:
            return False
        if value is not None and tok.value.upper() != value.upper():
            return False
        return True

    def _consume(self, ttype: TType, value: Optional[str] = None) -> Optional[Token]:
        if self._match(ttype, value):
            return self._advance()
        return None

    def _parse_identifier(self) -> str:
        tok = self._advance()
        if tok.ttype not in (TType.IDENTIFIER, TType.KEYWORD):
            raise ParseError(f"Expected identifier, got {tok}")
        return tok.value

    # ------------------------------------------------------------------
    # Top-level dispatch
    # ------------------------------------------------------------------

    def parse(self) -> Stmt:
        tok = self._peek()
        if tok.ttype == TType.KEYWORD:
            kw = tok.value.upper()
            if kw == "SELECT":
                return self._parse_select()
            elif kw == "INSERT":
                return self._parse_insert()
            elif kw == "UPDATE":
                return self._parse_update()
            elif kw == "DELETE":
                return self._parse_delete()
            elif kw == "CREATE":
                return self._parse_create()
            elif kw == "DROP":
                return self._parse_drop()
            elif kw in ("BEGIN", "START"):
                self._advance()
                self._consume_keyword("TRANSACTION")
                return BeginStmt()
            elif kw == "COMMIT":
                self._advance()
                return CommitStmt()
            elif kw == "ROLLBACK":
                self._advance()
                return RollbackStmt()
        raise ParseError(f"Unexpected token: {tok}")

    # ------------------------------------------------------------------
    # SELECT
    # ------------------------------------------------------------------

    def _parse_select(self) -> SelectStmt:
        self._expect_keyword("SELECT")
        distinct = bool(self._consume_keyword("DISTINCT"))

        # Column list
        columns: List[str] = []
        while True:
            if self._match(TType.STAR):
                self._advance()
                columns.append("*")
            else:
                col = self._parse_expr_text()
                if self._consume_keyword("AS"):
                    alias = self._parse_identifier()
                    col = f"{col} AS {alias}"
                columns.append(col)
            if not self._consume(TType.COMMA):
                break

        # FROM
        from_table: Optional[str] = None
        table_alias: Optional[str] = None
        joins: List[JoinClause] = []
        if self._consume_keyword("FROM"):
            from_table = self._parse_identifier()
            if self._consume_keyword("AS"):
                table_alias = self._parse_identifier()
            elif self._match(TType.IDENTIFIER):
                table_alias = self._parse_identifier()

            # JOINs
            while self._match_keyword("JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL"):
                joins.append(self._parse_join())

        # WHERE
        where: Optional[Expr] = None
        if self._consume_keyword("WHERE"):
            where = Expr(self._parse_condition_text())

        # GROUP BY
        group_by: List[str] = []
        if self._consume_keyword("GROUP"):
            self._expect_keyword("BY")
            group_by.append(self._parse_expr_text())
            while self._consume(TType.COMMA):
                group_by.append(self._parse_expr_text())

        # HAVING
        having: Optional[Expr] = None
        if self._consume_keyword("HAVING"):
            having = Expr(self._parse_condition_text())

        # ORDER BY
        order_by: List[Tuple[str, str]] = []
        if self._consume_keyword("ORDER"):
            self._expect_keyword("BY")
            while True:
                col = self._parse_order_col()
                direction = "ASC"
                if self._consume_keyword("DESC"):
                    direction = "DESC"
                elif self._consume_keyword("ASC"):
                    direction = "ASC"
                order_by.append((col, direction))
                if not self._consume(TType.COMMA):
                    break

        # LIMIT / OFFSET
        limit: Optional[int] = None
        offset: Optional[int] = None
        if self._consume_keyword("LIMIT"):
            tok = self._expect(TType.NUMBER)
            limit = int(tok.value)
        if self._consume_keyword("OFFSET"):
            tok = self._expect(TType.NUMBER)
            offset = int(tok.value)

        return SelectStmt(
            columns=columns,
            from_table=from_table,
            table_alias=table_alias,
            joins=joins,
            where=where,
            order_by=order_by,
            limit=limit,
            offset=offset,
            distinct=distinct,
            group_by=group_by,
            having=having,
        )

    def _parse_join(self) -> JoinClause:
        join_type = "INNER"
        if self._consume_keyword("INNER"):
            self._consume_keyword("JOIN")
        elif self._consume_keyword("LEFT"):
            self._consume_keyword("OUTER")
            self._consume_keyword("JOIN")
            join_type = "LEFT"
        elif self._consume_keyword("RIGHT"):
            self._consume_keyword("OUTER")
            self._consume_keyword("JOIN")
            join_type = "RIGHT"
        elif self._consume_keyword("FULL"):
            self._consume_keyword("OUTER")
            self._consume_keyword("JOIN")
            join_type = "FULL"
        elif self._consume_keyword("CROSS"):
            self._consume_keyword("JOIN")
            join_type = "CROSS"
        elif self._consume_keyword("NATURAL"):
            self._consume_keyword("JOIN")
            join_type = "NATURAL"
        else:
            self._consume_keyword("JOIN")

        table = self._parse_identifier()
        alias: Optional[str] = None
        if self._consume_keyword("AS"):
            alias = self._parse_identifier()
        elif self._match(TType.IDENTIFIER):
            alias = self._parse_identifier()

        condition: Optional[Expr] = None
        if self._consume_keyword("ON"):
            condition = Expr(self._parse_condition_text())

        return JoinClause(join_type, table, alias, condition)

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------

    def _parse_insert(self) -> InsertStmt:
        self._expect_keyword("INSERT")
        self._expect_keyword("INTO")
        table = self._parse_identifier()
        self._expect(TType.LPAREN)
        columns: List[str] = []
        while True:
            columns.append(self._parse_identifier())
            if not self._consume(TType.COMMA):
                break
        self._expect(TType.RPAREN)
        self._expect_keyword("VALUES")
        all_values: List[List[Any]] = []
        while True:
            self._expect(TType.LPAREN)
            row: List[Any] = []
            while True:
                row.append(self._parse_value())
                if not self._consume(TType.COMMA):
                    break
            self._expect(TType.RPAREN)
            all_values.append(row)
            if not self._consume(TType.COMMA):
                break
        return InsertStmt(table, columns, all_values)

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------

    def _parse_update(self) -> UpdateStmt:
        self._expect_keyword("UPDATE")
        table = self._parse_identifier()
        self._expect_keyword("SET")
        assignments: Dict[str, Any] = {}
        while True:
            col = self._parse_identifier()
            self._expect(TType.OPERATOR, "=")
            val = self._parse_value()
            assignments[col] = val
            if not self._consume(TType.COMMA):
                break
        where: Optional[Expr] = None
        if self._consume_keyword("WHERE"):
            where = Expr(self._parse_condition_text())
        return UpdateStmt(table, assignments, where)

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def _parse_delete(self) -> DeleteStmt:
        self._expect_keyword("DELETE")
        self._expect_keyword("FROM")
        table = self._parse_identifier()
        where: Optional[Expr] = None
        if self._consume_keyword("WHERE"):
            where = Expr(self._parse_condition_text())
        return DeleteStmt(table, where)

    # ------------------------------------------------------------------
    # CREATE TABLE / INDEX
    # ------------------------------------------------------------------

    def _parse_create(self) -> Stmt:
        self._expect_keyword("CREATE")
        unique = bool(self._consume_keyword("UNIQUE"))
        if self._consume_keyword("INDEX"):
            return self._parse_create_index(unique)
        self._expect_keyword("TABLE")
        if_not_exists = False
        if self._consume_keyword("IF"):
            self._expect_keyword("NOT")
            self._expect_keyword("EXISTS")
            if_not_exists = True
        table = self._parse_identifier()
        self._expect(TType.LPAREN)
        columns: List[ColumnSpec] = []
        while True:
            columns.append(self._parse_column_spec())
            if not self._consume(TType.COMMA):
                break
        self._expect(TType.RPAREN)
        return CreateTableStmt(table, columns, if_not_exists)

    def _parse_column_spec(self) -> ColumnSpec:
        name = self._parse_identifier()
        col_type = self._parse_identifier()
        nullable = True
        primary_key = False
        default = None
        unique = False
        while self._match_keyword("NOT", "PRIMARY", "DEFAULT", "UNIQUE", "NULL"):
            if self._consume_keyword("NOT"):
                self._consume_keyword("NULL")
                nullable = False
            elif self._consume_keyword("PRIMARY"):
                self._consume_keyword("KEY")
                primary_key = True
                nullable = False
            elif self._consume_keyword("UNIQUE"):
                unique = True
            elif self._consume_keyword("DEFAULT"):
                default = str(self._parse_value())
            else:
                break
        return ColumnSpec(name, col_type, nullable, primary_key, default, unique)

    def _parse_create_index(self, unique: bool) -> CreateIndexStmt:
        index_name = self._parse_identifier()
        self._expect_keyword("ON")
        table = self._parse_identifier()
        self._expect(TType.LPAREN)
        cols: List[str] = []
        while True:
            cols.append(self._parse_identifier())
            if not self._consume(TType.COMMA):
                break
        self._expect(TType.RPAREN)
        return CreateIndexStmt(index_name, table, cols, unique)

    # ------------------------------------------------------------------
    # DROP
    # ------------------------------------------------------------------

    def _parse_drop(self) -> Stmt:
        self._expect_keyword("DROP")
        if self._consume_keyword("INDEX"):
            name = self._parse_identifier()
            return DropIndexStmt(name)
        self._expect_keyword("TABLE")
        if_exists = False
        if self._consume_keyword("IF"):
            self._expect_keyword("EXISTS")
            if_exists = True
        table = self._parse_identifier()
        return DropTableStmt(table, if_exists)

    # ------------------------------------------------------------------
    # Expression helpers
    # ------------------------------------------------------------------

    def _parse_value(self) -> Any:
        tok = self._peek()
        if tok.ttype == TType.STRING:
            self._advance()
            return tok.value
        elif tok.ttype == TType.NUMBER:
            self._advance()
            v = tok.value
            return float(v) if "." in v else int(v)
        elif tok.ttype == TType.KEYWORD and tok.value.upper() == "NULL":
            self._advance()
            return None
        elif tok.ttype == TType.KEYWORD and tok.value.upper() in ("TRUE", "FALSE"):
            self._advance()
            return tok.value.upper() == "TRUE"
        elif tok.ttype in (TType.IDENTIFIER, TType.KEYWORD):
            # Could be a function call or bare identifier — return as string expr
            name = self._advance().value
            if self._match(TType.LPAREN):
                # function call — consume args
                self._advance()
                args = []
                while not self._match(TType.RPAREN):
                    if self._match(TType.COMMA):
                        self._advance()
                        continue
                    args.append(str(self._parse_value()))
                self._advance()
                return f"{name}({', '.join(args)})"
            return name
        raise ParseError(f"Expected value, got {tok}")

    def _parse_order_col(self) -> str:
        """Collect a column/expression for ORDER BY, stopping before ASC/DESC."""
        STOP = {
            "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS",
            "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET", "ON", "AS",
            "AND", "OR", "SET", "VALUES", "INTO", "ASC", "DESC",
        }
        parts = []
        depth = 0
        while True:
            tok = self._peek()
            if tok.ttype == TType.EOF:
                break
            if tok.ttype == TType.LPAREN:
                depth += 1
            elif tok.ttype == TType.RPAREN:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if tok.ttype == TType.COMMA:
                    break
                if tok.ttype == TType.KEYWORD and tok.value.upper() in STOP:
                    break
            parts.append(tok.value)
            self._advance()
        return " ".join(parts)

    def _parse_expr_text(self) -> str:
        """Collect tokens for a single expression (until comma/keyword sentinel)."""
        STOP = {
            "FROM", "WHERE", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS",
            "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET", "ON", "AS",
            "AND", "OR", "SET", "VALUES", "INTO",
        }
        parts = []
        depth = 0
        while True:
            tok = self._peek()
            if tok.ttype == TType.EOF:
                break
            if tok.ttype == TType.LPAREN:
                depth += 1
            elif tok.ttype == TType.RPAREN:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if tok.ttype == TType.COMMA:
                    break
                if tok.ttype == TType.KEYWORD and tok.value.upper() in STOP:
                    break
            parts.append(tok.value)
            self._advance()
        return " ".join(parts)

    def _parse_condition_text(self) -> str:
        """Collect tokens for a WHERE/HAVING/ON condition."""
        STOP = {
            "ORDER", "GROUP", "HAVING", "LIMIT", "OFFSET",
            "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL",
            "RETURNING", "SET",
        }
        parts = []
        depth = 0
        while True:
            tok = self._peek()
            if tok.ttype == TType.EOF:
                break
            if tok.ttype == TType.LPAREN:
                depth += 1
            elif tok.ttype == TType.RPAREN:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if tok.ttype in (TType.SEMICOLON,):
                    break
                if tok.ttype == TType.KEYWORD and tok.value.upper() in STOP:
                    break
            parts.append(tok.value)
            self._advance()
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_sql(sql: str) -> Stmt:
    """Parse a SQL string and return an AST node."""
    tokens = tokenize(sql.strip().rstrip(";"))
    return Parser(tokens).parse()
