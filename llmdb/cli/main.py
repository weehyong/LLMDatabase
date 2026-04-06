"""Command-line interface for LLMDatabase."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from llmdb.engine import Database


def _open_db(db_path: str) -> Database:
    return Database(db_path)


@click.group()
@click.option(
    "--db",
    default="llmdb.db",
    show_default=True,
    envvar="LLMDB_PATH",
    help="Path to the database file.",
)
@click.pass_context
def cli(ctx: click.Context, db: str) -> None:
    """LLMDatabase — a database for agentic applications."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


# ---------------------------------------------------------------------------
# Collection commands
# ---------------------------------------------------------------------------


@cli.group("collection")
def collection_group():
    """Manage collections."""


@collection_group.command("create")
@click.argument("name")
@click.option("--vector-dim", type=int, default=None, help="Embedding dimension.")
@click.option("--exist-ok", is_flag=True, help="Don't error if already exists.")
@click.pass_context
def collection_create(ctx, name, vector_dim, exist_ok):
    """Create a new collection."""
    with _open_db(ctx.obj["db_path"]) as db:
        coll = db.create_collection(name, vector_dim=vector_dim, exist_ok=exist_ok)
        click.echo(f"✓ Created collection '{coll.name}' (vector_dim={coll.vector_dim})")


@collection_group.command("list")
@click.pass_context
def collection_list(ctx):
    """List all collections."""
    with _open_db(ctx.obj["db_path"]) as db:
        colls = db.list_collections()
    if not colls:
        click.echo("No collections found.")
        return
    click.echo(f"{'Name':<30} {'Docs':>8} {'Vector dim':>12}  Created")
    click.echo("-" * 70)
    for c in colls:
        click.echo(
            f"{c['name']:<30} {'—':>8} {str(c['vector_dim'] or '—'):>12}  {c['created_at']}"
        )


@collection_group.command("drop")
@click.argument("name")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.pass_context
def collection_drop(ctx, name, yes):
    """Drop a collection and all its documents."""
    if not yes:
        click.confirm(f"Drop collection '{name}' and all its documents?", abort=True)
    with _open_db(ctx.obj["db_path"]) as db:
        db.drop_collection(name)
    click.echo(f"✓ Dropped collection '{name}'.")


# ---------------------------------------------------------------------------
# Document commands
# ---------------------------------------------------------------------------


@cli.group("doc")
def doc_group():
    """Manage documents."""


@doc_group.command("insert")
@click.argument("collection")
@click.argument("data")
@click.option("--tags", default="", help="Comma-separated tags.")
@click.option("--score", type=float, default=0.0)
@click.pass_context
def doc_insert(ctx, collection, data, tags, score):
    """Insert a document (DATA must be a JSON string)."""
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON: {exc}", err=True)
        sys.exit(1)
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    with _open_db(ctx.obj["db_path"]) as db:
        doc = db.get_collection(collection).insert(
            payload, tags=tag_list, score=score
        )
    click.echo(json.dumps(doc.to_dict(), indent=2))


@doc_group.command("get")
@click.argument("collection")
@click.argument("doc_id")
@click.pass_context
def doc_get(ctx, collection, doc_id):
    """Retrieve a document by ID."""
    with _open_db(ctx.obj["db_path"]) as db:
        doc = db.get(collection, doc_id)
    click.echo(json.dumps(doc.to_dict(), indent=2))


@doc_group.command("list")
@click.argument("collection")
@click.option("--tags", default="", help="Comma-separated tags to filter by.")
@click.option("--limit", type=int, default=20)
@click.pass_context
def doc_list(ctx, collection, tags, limit):
    """List documents in a collection."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    with _open_db(ctx.obj["db_path"]) as db:
        docs = db.get_collection(collection).list(
            tags=tag_list or None, limit=limit
        )
    click.echo(json.dumps([d.to_dict() for d in docs], indent=2))


@doc_group.command("delete")
@click.argument("collection")
@click.argument("doc_id")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
@click.pass_context
def doc_delete(ctx, collection, doc_id, yes):
    """Delete a document."""
    if not yes:
        click.confirm(f"Delete document '{doc_id}'?", abort=True)
    with _open_db(ctx.obj["db_path"]) as db:
        db.delete(collection, doc_id)
    click.echo(f"✓ Deleted document '{doc_id}'.")


@doc_group.command("query")
@click.argument("collection")
@click.argument("filter_json", default="{}")
@click.option("--limit", type=int, default=20)
@click.option("--sort-by", default=None)
@click.pass_context
def doc_query(ctx, collection, filter_json, limit, sort_by):
    """Run a structured query (FILTER_JSON must be a JSON filter dict)."""
    try:
        filt = json.loads(filter_json) or None
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON: {exc}", err=True)
        sys.exit(1)
    with _open_db(ctx.obj["db_path"]) as db:
        docs = db.query(collection, filt, sort_by=sort_by, limit=limit)
    click.echo(json.dumps([d.to_dict() for d in docs], indent=2))


# ---------------------------------------------------------------------------
# Memory commands
# ---------------------------------------------------------------------------


@cli.group("memory")
def memory_group():
    """Manage agent memory."""


@memory_group.command("add")
@click.argument("content")
@click.option(
    "--type",
    "memory_type",
    default="episodic",
    type=click.Choice(["episodic", "semantic", "procedural"]),
)
@click.option("--importance", type=float, default=0.5)
@click.option("--tags", default="", help="Comma-separated tags.")
@click.pass_context
def memory_add(ctx, content, memory_type, importance, tags):
    """Add a new memory."""
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    with _open_db(ctx.obj["db_path"]) as db:
        mem = db.memory.add(
            content,
            memory_type=memory_type,
            importance=importance,
            tags=tag_list,
        )
    click.echo(json.dumps(mem.to_dict(), indent=2, default=str))


@memory_group.command("list")
@click.option(
    "--type",
    "memory_type",
    default=None,
    type=click.Choice(["episodic", "semantic", "procedural"]),
)
@click.option("--limit", type=int, default=20)
@click.option("--min-importance", type=float, default=0.0)
@click.pass_context
def memory_list(ctx, memory_type, limit, min_importance):
    """List memories, ordered by importance."""
    with _open_db(ctx.obj["db_path"]) as db:
        mems = db.memory.list(
            memory_type=memory_type, min_importance=min_importance, limit=limit
        )
    click.echo(json.dumps([m.to_dict() for m in mems], indent=2, default=str))


@memory_group.command("delete")
@click.argument("memory_id")
@click.option("--yes", is_flag=True)
@click.pass_context
def memory_delete(ctx, memory_id, yes):
    """Delete a memory by ID."""
    if not yes:
        click.confirm(f"Delete memory '{memory_id}'?", abort=True)
    with _open_db(ctx.obj["db_path"]) as db:
        db.memory.delete(memory_id)
    click.echo(f"✓ Deleted memory '{memory_id}'.")


# ---------------------------------------------------------------------------
# Server command
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8000, show_default=True)
@click.pass_context
def serve(ctx, host, port):
    """Start the REST API server."""
    from llmdb.api.server import run

    click.echo(f"Starting LLMDatabase server on http://{host}:{port}")
    run(db_path=ctx.obj["db_path"], host=host, port=port)


if __name__ == "__main__":
    cli()
