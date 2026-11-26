"""Benchmark several Ollama model/quantization pairs on the scanned split.

For each configuration:

1. One warmup LLM call on the first document (excluded from metrics).
2. ``--repeats`` passes over the 20 non-dev scanned documents, measuring the
   LLM latency of each call.
3. Memory of the loaded model is read from Ollama's ``/api/ps`` endpoint.
4. One raw-response probe per model is captured separately (``--probe-only``)
   so the user can inspect format quirks (reasoning blocks, markdown
   fences, etc.) before investing in the full run.

Writes ``eval/reports/benchmark_<date>.md`` and ``eval/results/benchmark_<date>.jsonl``.

Usage::

    # Format probe only: one call per model, raw text dumped
    uv run python eval/benchmark.py --models tag1 tag2 --probe-only

    # Full benchmark (default 3 repeats on 20 non-dev scanned docs)
    uv run python eval/benchmark.py --models tag1 tag2 [--repeats 3]

The LLM call uses the project's own ``LLMClient``; no service code is
modified.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

from evaluate import (  # noqa: E402
    ALL_FIELDS,
    GT_DIR,
    HELDOUT_DEV_IDS,
    REPORTS_DIR,
    RESULTS_DIR,
    SCHEMA_PATH,
    _check_ollama,
    _load_manifest,
    _percentile,
    _run_llm_once,
    score_document,
)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------


def _get_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_ps() -> list[dict]:
    """Return ``/api/ps`` model list; empty if nothing is loaded."""
    try:
        return _get_json("http://127.0.0.1:11434/api/ps").get("models", [])
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        sys.stderr.write(f"WARN: ollama /api/ps unreachable ({e}); memory column will be 0.\n")
        return []


def read_model_memory(tag: str) -> dict:
    """Snapshot memory for ``tag`` from ``/api/ps``.

    Returns ``{size}`` — Ollama's self-reported footprint (weights + KV cache)
    in bytes. Not an OS-level RSS; the caveat is surfaced in the report.
    """
    for m in ollama_ps():
        if m.get("name") == tag or m.get("model") == tag:
            return {"size": m.get("size", 0)}
    return {"size": 0}


def _bytes_to_gb(n: int) -> float:
    return n / (1024 ** 3)


# ---------------------------------------------------------------------------
# Raw format probe
# ---------------------------------------------------------------------------


async def raw_probe(model_tag: str, markdown: str, schema_yaml: str) -> dict:
    """Make one chat.completions call with JSON mode and return the raw content.

    Unlike ``LLMClient.request`` this does not parse or validate the reply;
    it simply surfaces whatever the model produced, including empty content
    (e.g. when a model routes reasoning into a sibling field).
    """
    from openai import AsyncOpenAI
    from client import SYSTEM_PROMPT

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1", api_key="ollama_dummy_key"
    )
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_tag,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(schema_yaml=schema_yaml)},
            {"role": "user", "content": markdown},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    latency = time.perf_counter() - start
    choice = response.choices[0]
    msg = choice.message
    return {
        "model": model_tag,
        "latency_s": round(latency, 2),
        "finish_reason": choice.finish_reason,
        "content": msg.content,
        "content_len": len(msg.content or ""),
        # Capture any non-standard fields the server may return (reasoning).
        "extra_message_fields": {
            k: v
            for k, v in msg.model_dump(exclude_none=False).items()
            if k not in ("role", "content", "refusal") and v is not None
        },
    }


# ---------------------------------------------------------------------------
# Per-model benchmark run
# ---------------------------------------------------------------------------


async def bench_one(
    model_tag: str,
    docs: list[tuple[str, int, str]],  # (doc_id, template, markdown)
    schema_yaml: str,
    schema_type,
    repeats: int,
) -> dict:
    """Run warmup + ``repeats`` passes over ``docs`` for one model."""
    _check_ollama(model_tag)
    os.environ["OLLAMA_MODEL"] = model_tag

    from client import LLMClient

    llm = LLMClient()

    # Warmup — first doc, excluded from stats.
    warm_doc_id, _, warm_md = docs[0]
    _, warm_lat, warm_err = await _run_llm_once(llm, warm_md, schema_yaml, schema_type)
    sys.stdout.write(
        f"  [warmup] {model_tag}  on {warm_doc_id}  llm={warm_lat:.1f}s  err={warm_err}\n"
    )

    memory = read_model_memory(model_tag)

    per_doc_rows: list[dict] = []
    for rep in range(repeats):
        for doc_id, template, md in docs:
            gt = json.loads((GT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))
            pred, latency, err = await _run_llm_once(llm, md, schema_yaml, schema_type)
            scores = score_document(gt, pred)
            per_doc_rows.append(
                {
                    "model": model_tag,
                    "rep": rep,
                    "doc_id": doc_id,
                    "template": template,
                    "llm_s": latency,
                    "error": err,
                    "prediction": pred,
                    "scores": {
                        k: {"strict": v.strict, "normalized": v.normalized}
                        for k, v in scores.items()
                    },
                }
            )
            tag = "OK " if pred is not None else "ERR"
            sys.stdout.write(
                f"  [{tag}] {model_tag}  rep{rep+1}  T{template} {doc_id}  "
                f"llm={latency:.1f}s\n"
            )
            sys.stdout.flush()

    return {"rows": per_doc_rows, "memory": memory, "warmup_latency_s": warm_lat}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg(rows: list[dict], predicate=lambda r: True) -> dict:
    sel = [r for r in rows if predicate(r)]
    if not sel:
        return {}
    n = len(sel)
    invalid = sum(1 for r in sel if r["prediction"] is None)
    latencies = [r["llm_s"] for r in sel]
    per_field_norm: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    for r in sel:
        for f in ALL_FIELDS:
            per_field_norm[f].append(r["scores"][f]["normalized"])
    return {
        "n": n,
        "invalid_pct": 100.0 * invalid / n,
        "latency_median_s": statistics.median(latencies),
        "latency_p90_s": _percentile(latencies, 90),
        "per_field_norm": {
            f: {
                "mean": sum(xs) / n,
                "perfect": sum(1 for x in xs if x == 1.0),
            }
            for f, xs in per_field_norm.items()
        },
    }


def _mean_norm(agg: dict) -> float:
    if not agg:
        return 0.0
    return sum(v["mean"] for v in agg["per_field_norm"].values()) / len(ALL_FIELDS)


def _render_report(
    benches: list[dict], n_unique_docs: int, repeats: int
) -> str:
    lines: list[str] = []
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"# Ollama model benchmark — {stamp}")
    lines.append("")
    lines.append(
        f"- Sample: {n_unique_docs} scanned documents (non-dev), "
        f"{repeats} repetitions each = {n_unique_docs * repeats} LLM calls per config."
    )
    lines.append(
        "- Field scores are the **normalized** mean across all calls for that "
        "config. T3 columns are the same metric restricted to template-3 docs."
    )
    lines.append(
        "- Memory is Ollama's `/api/ps` `size` for the loaded model (weights + KV). "
        "It is self-reported, not an OS RSS of the server process."
    )
    lines.append(
        f"- Caveat: n={n_unique_docs} unique docs is small; "
        "field scores at this sample size have ±5–10 pts of noise."
    )
    lines.append("")

    # ------------ Headline table ------------
    lines.append("## Per-configuration summary")
    lines.append("")
    lines.append(
        "| Model | Mem (GB) | Mean norm | T3 mean norm | Invalid JSON | Lat. median (s) | Lat. p90 (s) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in benches:
        all_rows = b["rows"]
        t3_rows = [r for r in all_rows if r["template"] == 3]
        agg = _agg(all_rows)
        agg_t3 = _agg(t3_rows)
        if not agg:
            lines.append(
                f"| {b['tag']} | — | — | — | — | — | — |"
            )
            continue
        mem_gb = _bytes_to_gb(b["memory"]["size"])
        lines.append(
            f"| `{b['tag']}` | {mem_gb:.2f} | "
            f"{_mean_norm(agg):.2f} | "
            f"{_mean_norm(agg_t3):.2f} ({len(t3_rows)} calls) | "
            f"{agg['invalid_pct']:.1f}% | "
            f"{agg['latency_median_s']:.1f} | "
            f"{agg['latency_p90_s']:.1f} |"
        )
    lines.append("")

    # ------------ Per-field table per config ------------
    lines.append("## Normalized score per field (mean over all calls)")
    lines.append("")
    header_mods = [b["tag"] for b in benches]
    lines.append("| Field | " + " | ".join(f"`{m}`" for m in header_mods) + " |")
    lines.append("|---" + "|---:" * len(header_mods) + "|")
    aggs = [_agg(b["rows"]) for b in benches]
    for f in ALL_FIELDS:
        row = [f]
        for a in aggs:
            if not a:
                row.append("—")
                continue
            v = a["per_field_norm"][f]
            row.append(f"{v['mean']:.2f} ({v['perfect']}/{a['n']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ------------ Quantization effect per family ------------
    lines.append("## Quantization effect per family (Q4_K_M vs Q8_0)")
    lines.append("")
    lines.append("| Family | Q4 mean norm | Q8 mean norm | Δ (Q8−Q4) | Q4 lat. med. | Q8 lat. med. |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    families: dict[str, dict[str, dict]] = {}
    for b in benches:
        tag = b["tag"]
        family = tag.split(":")[0]
        quant = (
            "Q4" if "q4" in tag.lower() else ("Q8" if "q8" in tag.lower() else "other")
        )
        families.setdefault(family, {})[quant] = _agg(b["rows"])
    for family, by_q in families.items():
        q4 = by_q.get("Q4", {})
        q8 = by_q.get("Q8", {})
        if not q4 or not q8:
            lines.append(f"| {family} | — | — | — | — | — |")
            continue
        q4m, q8m = _mean_norm(q4), _mean_norm(q8)
        lines.append(
            f"| {family} | {q4m:.2f} | {q8m:.2f} | {q8m - q4m:+.2f} | "
            f"{q4['latency_median_s']:.1f}s | {q8['latency_median_s']:.1f}s |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_non_dev_scanned() -> list[tuple[str, int, str]]:
    """Return ``(doc_id, template, cached_markdown)`` for every non-dev scanned doc."""
    manifest = _load_manifest()
    docs = []
    for doc_id in sorted(manifest.keys()):
        if doc_id in HELDOUT_DEV_IDS:
            continue
        cache_file = ROOT / "cache" / "scanned" / f"{doc_id}.json"
        if not cache_file.exists():
            raise SystemExit(
                f"Missing docling cache for {doc_id} (scanned). "
                f"Run `uv run python eval/evaluate.py --model <any> --split scanned` once to warm."
            )
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        docs.append((doc_id, int(manifest[doc_id]["template"]), cached["markdown"]))
    return docs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True, help="Ollama tags to bench.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--probe-only",
        action="store_true",
        help="Make a single raw-response call per model and dump it, no scoring.",
    )
    return p.parse_args(argv)


async def async_main(args):
    import yaml
    from models import build_dynamic_model

    schema_yaml = SCHEMA_PATH.read_text(encoding="utf-8")
    schema_type = build_dynamic_model(yaml.safe_load(schema_yaml))
    docs = _load_non_dev_scanned()

    if args.probe_only:
        # One raw call per model on the first doc.
        _, _, md = docs[0]
        for tag in args.models:
            _check_ollama(tag)
            probe = await raw_probe(tag, md, schema_yaml)
            print(f"\n==================== {tag} ====================")
            print(
                f"finish_reason={probe['finish_reason']}  "
                f"content_len={probe['content_len']}  latency={probe['latency_s']}s"
            )
            if probe["extra_message_fields"]:
                print("Non-standard message fields:")
                print(json.dumps(probe["extra_message_fields"], indent=2)[:4000])
            print("--- content ---")
            print((probe["content"] or "<empty>")[:4000])
        return

    benches = []
    for tag in args.models:
        print(f"\n### {tag}")
        b = await bench_one(tag, docs, schema_yaml, schema_type, args.repeats)
        b["tag"] = tag
        benches.append(b)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl = RESULTS_DIR / f"benchmark_{stamp}.jsonl"
    md = REPORTS_DIR / f"benchmark_{stamp}.md"
    with jsonl.open("w", encoding="utf-8") as f:
        for b in benches:
            for row in b["rows"]:
                f.write(json.dumps({**row, "memory": b["memory"]}, ensure_ascii=False) + "\n")
    md.write_text(_render_report(benches, len(docs), args.repeats), encoding="utf-8")
    print(f"\nWrote {jsonl}")
    print(f"Wrote {md}")


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
