"""Field-by-field extraction evaluation for the SIE service.

Usage::

    uv run python eval/evaluate.py --model <ollama_tag> \
        [--split scanned|clean|all] [--limit N] [--repeat K]

What it does:

1. Caches the docling (Markdown) output of every PDF the first time it is
   seen, under ``eval/cache/<split>/<doc_id>.json`` (markdown + docling
   latency). Subsequent runs reuse the cache so only the LLM step is
   measured during a benchmark.
2. Calls the LLM step (``LLMClient.request``) on each cached Markdown, with
   the Ollama tag given by ``--model`` (exported as ``OLLAMA_MODEL``).
3. Scores each document field-by-field, with a strict and a normalized
   variant, and writes:
   - ``eval/results/<model>_<date>.jsonl`` — one JSON line per document.
   - ``eval/reports/<model>_<date>.md`` — summary tables per split.
4. ``--repeat K`` reruns each document in the limited subset K times so the
   variance of the LLM latency can be inspected.

This module does NOT touch the FastAPI service or Extractor/LLMClient code;
it imports them directly.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import hashlib
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importing extractor is slow (torch + docling). Deferred until first use.
_extractor = None


def _get_extractor():
    """Lazily instantiate the shared docling Extractor (slow at startup)."""
    global _extractor
    if _extractor is None:
        from extractor import Extractor  # noqa: WPS433 — intentional lazy import.

        _extractor = Extractor()
    return _extractor


CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
GT_DIR = ROOT / "ground_truth"
MANIFEST = ROOT / "manifest.csv"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "base.yaml"

SCALAR_FIELDS = [
    "company_name",
    "bl_number",
    "shipper",
    "port_of_loading",
    "port_of_discharge",
    "place_of_acceptance",
    "place_of_delivery",
    "shipped_on_board_date",
    "vessel_name",
    "voyage_number",
    "mark",
]
LIST_FIELDS = ["container_numbers", "lots"]
ALL_FIELDS = SCALAR_FIELDS + LIST_FIELDS


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w]+", flags=re.UNICODE)

_DATE_INPUT_FORMATS = [
    "%d %b %Y",      # 03 JUN 2026
    "%d %B %Y",      # 03 June 2026
    "%Y-%m-%d",      # 2026-06-03
    "%d/%m/%Y",      # 03/06/2026
    "%Y/%m/%d",      # 2026/06/03
    "%B %d, %Y",     # June 3, 2026
    "%B %d %Y",      # June 3 2026
    "%b %d, %Y",     # Jun 3, 2026
    "%b %d %Y",      # Jun 3 2026
    "%d-%m-%Y",      # 03-06-2026
    "%Y%m%d",        # 20260603
    "%d.%m.%Y",      # 03.06.2026
]

_WEIGHT_RE = re.compile(
    r"\b(\d{1,3}(?:[,\s]\d{3})*|\d+)(?:\.\d+)?\s*(kgs?|kilograms?|lbs?|pounds?)\b",
    flags=re.IGNORECASE,
)


def _canonicalize_date(value: str) -> str | None:
    """Return ``YYYY-MM-DD`` if ``value`` parses as any known date format."""
    v = value.strip().rstrip(".")
    for fmt in _DATE_INPUT_FORMATS:
        try:
            return datetime.datetime.strptime(v, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _canonicalize_weights(text: str) -> str:
    """Rewrite each inline weight to a canonical ``<integer-kg> kg`` form."""

    def _sub(match: re.Match) -> str:
        num = match.group(1).replace(",", "").replace(" ", "")
        try:
            value = float(num)
        except ValueError:
            return match.group(0)
        unit = match.group(2).lower()
        if unit.startswith("lb") or unit.startswith("pound"):
            value *= 0.45359237
        return f"{int(round(value))} kg"

    return _WEIGHT_RE.sub(_sub, text)


def normalize_scalar(field_name: str, value) -> str:
    """Normalize a string value for the "normalized" comparison.

    - ``shipped_on_board_date``: parse to ``YYYY-MM-DD`` when possible.
    - Any other field: canonicalize inline weights, lowercase, collapse all
      non-word characters (incl. spaces and punctuation) to an empty string.
    """
    if value is None:
        return ""
    s = str(value)
    if field_name == "shipped_on_board_date":
        canonical = _canonicalize_date(s)
        if canonical is not None:
            return canonical
    s = _canonicalize_weights(s)
    s = s.lower()
    s = _PUNCT_RE.sub("", s)
    return s


def normalize_container(value) -> str:
    """Normalize a container number: uppercase, strip non-alphanumerics."""
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class FieldScore:
    strict: float
    normalized: float


def _score_scalar(expected, predicted, field_name: str) -> FieldScore:
    exp_strict = "" if expected is None else str(expected)
    pred_strict = "" if predicted is None else str(predicted)
    strict = 1.0 if exp_strict == pred_strict else 0.0

    exp_norm = normalize_scalar(field_name, expected)
    pred_norm = normalize_scalar(field_name, predicted)
    normalized = 1.0 if exp_norm and exp_norm == pred_norm else 0.0
    return FieldScore(strict=strict, normalized=normalized)


def _f1(expected: set, predicted: set) -> float:
    if not expected and not predicted:
        return 1.0
    if not predicted or not expected:
        return 0.0
    tp = len(expected & predicted)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall)


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _score_containers(expected, predicted) -> FieldScore:
    exp = _as_list(expected)
    pred = _as_list(predicted)
    strict = _f1(set(map(str, exp)), set(map(str, pred)))
    normalized = _f1(
        {normalize_container(v) for v in exp if v is not None},
        {normalize_container(v) for v in pred if v is not None},
    )
    return FieldScore(strict=strict, normalized=normalized)


def _score_lots(expected, predicted) -> FieldScore:
    exp = _as_list(expected)
    pred = _as_list(predicted)
    strict = _f1(set(map(str, exp)), set(map(str, pred)))
    normalized = _f1(
        {normalize_scalar("lots", v) for v in exp if v is not None},
        {normalize_scalar("lots", v) for v in pred if v is not None},
    )
    return FieldScore(strict=strict, normalized=normalized)


def score_document(expected: dict, predicted: dict | None) -> dict[str, FieldScore]:
    """Score every schema field. ``predicted=None`` means a total failure
    (every field gets 0)."""
    scores: dict[str, FieldScore] = {}
    if predicted is None:
        return {f: FieldScore(0.0, 0.0) for f in ALL_FIELDS}
    for field_name in SCALAR_FIELDS:
        scores[field_name] = _score_scalar(
            expected.get(field_name), predicted.get(field_name), field_name
        )
    scores["container_numbers"] = _score_containers(
        expected.get("container_numbers"), predicted.get("container_numbers")
    )
    scores["lots"] = _score_lots(
        expected.get("lots"), predicted.get("lots")
    )
    return scores


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def ensure_cached(split: str, doc_id: str, pdf_path: Path) -> dict:
    """Return the cached docling output for ``pdf_path``, populating the
    cache if necessary.

    The cache is invalidated when the PDF's SHA-1 changes (e.g. after
    regenerating the dataset).
    """
    cache_file = CACHE_DIR / split / f"{doc_id}.json"
    sha1 = _sha1(pdf_path)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("source_sha1") == sha1:
            return cached
    extractor = _get_extractor()
    start = time.perf_counter()
    markdown = extractor.extract_from_bytes(pdf_path.read_bytes(), pdf_path.name)
    docling_s = time.perf_counter() - start
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "doc_id": doc_id,
        "split": split,
        "source_sha1": sha1,
        "docling_s": round(docling_s, 3),
        "cached_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "markdown": markdown,
    }
    cache_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


@dataclass
class DocResult:
    doc_id: str
    split: str  # "clean", "scanned-light", "scanned-heavy"
    template: int  # 1, 2 or 3
    llm_s: float | None
    prediction: dict | None
    scores: dict[str, FieldScore]
    error: str | None = None
    repeat_latencies: list[float] = field(default_factory=list)


# Fixed held-out set for prompt tuning: 10 docs, balanced over templates
# (3 T1, 3 T2, 4 T3) and degradation levels (5 heavy, 5 light). Kept here —
# not in manifest.csv — so that regenerating the dataset does not move it.
HELDOUT_DEV_IDS: frozenset[str] = frozenset(
    {
        "bl_003", "bl_006", "bl_024",  # T1 — H, L, L
        "bl_001", "bl_010", "bl_019",  # T2 — H, L, H
        "bl_005", "bl_008", "bl_023", "bl_026",  # T3 — H, L, H, L
    }
)


def _load_manifest() -> dict[str, dict]:
    with MANIFEST.open(encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def _select_docs(
    split: str,
    limit: int | None,
    manifest: dict,
    exclude_dev: bool = False,
    only_dev: bool = False,
) -> list[tuple[str, str, int, Path]]:
    """Yield ``(doc_id, derived_split, template, pdf_path)`` tuples.

    ``split`` is the CLI filter (``clean``, ``scanned``, ``all``) and
    ``derived_split`` is the reporting bucket (``clean``, ``scanned-light``,
    ``scanned-heavy``). ``exclude_dev`` skips the fixed held-out
    ``HELDOUT_DEV_IDS``; ``only_dev`` keeps only those docs (for prompt tuning).
    """
    picks: list[tuple[str, str, int, Path]] = []
    doc_ids = sorted(manifest.keys())
    for doc_id in doc_ids:
        if exclude_dev and doc_id in HELDOUT_DEV_IDS:
            continue
        if only_dev and doc_id not in HELDOUT_DEV_IDS:
            continue
        row = manifest[doc_id]
        template = int(row["template"])
        if split in ("clean", "all"):
            picks.append((doc_id, "clean", template, ROOT / "clean" / f"{doc_id}.pdf"))
        if split in ("scanned", "all"):
            bucket = f"scanned-{row['degrade_level']}"
            picks.append(
                (doc_id, bucket, template, ROOT / "scanned" / f"{doc_id}.pdf")
            )
    if limit is not None:
        picks = picks[:limit]
    return picks


async def _run_llm_once(llm_client, markdown: str, schema_yaml: str, schema_type):
    """Call the LLM once, returning (prediction_dict_or_None, latency_s, error)."""
    start = time.perf_counter()
    try:
        model = await llm_client.request(markdown, schema_yaml, schema_type)
        latency = time.perf_counter() - start
        return model.model_dump(), latency, None
    except Exception as e:
        latency = time.perf_counter() - start
        return None, latency, f"{type(e).__name__}: {e}"


def _check_ollama(model_tag: str) -> None:
    """Fail fast if Ollama is not reachable or the requested model is missing.

    Uses the native tags endpoint rather than the OpenAI-compatible one:
    ``/api/tags`` always lists every installed model, while the OpenAI
    endpoint can return aliases.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=5
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise SystemExit(
            f"Ollama is not reachable at http://127.0.0.1:11434 ({e}). "
            "Start it with `ollama serve`."
        )
    installed = [m["name"] for m in payload.get("models", [])]
    if model_tag not in installed:
        raise SystemExit(
            f"Model {model_tag!r} is not installed. Available: {installed}. "
            f"Install with `ollama pull {model_tag}`."
        )


async def evaluate(args: argparse.Namespace) -> list[DocResult]:
    import yaml
    from client import LLMClient
    from models import build_dynamic_model

    _check_ollama(args.model)
    os.environ["OLLAMA_MODEL"] = args.model
    llm_client = LLMClient()

    schema_yaml = SCHEMA_PATH.read_text(encoding="utf-8")
    schema_type = build_dynamic_model(yaml.safe_load(schema_yaml))

    manifest = _load_manifest()
    picks = _select_docs(
        args.split,
        args.limit,
        manifest,
        exclude_dev=args.split_dev,
        only_dev=args.only_dev,
    )

    results: list[DocResult] = []
    for doc_id, bucket, template, pdf_path in picks:
        cached = ensure_cached(bucket.split("-", 1)[0], doc_id, pdf_path)
        markdown = cached["markdown"]
        gt = json.loads((GT_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))

        prediction, latency, error = await _run_llm_once(
            llm_client, markdown, schema_yaml, schema_type
        )
        scores = score_document(gt, prediction)

        repeats: list[float] = []
        if args.repeat > 1:
            for _ in range(args.repeat - 1):
                _, lat, _ = await _run_llm_once(
                    llm_client, markdown, schema_yaml, schema_type
                )
                repeats.append(lat)

        results.append(
            DocResult(
                doc_id=doc_id,
                split=bucket,
                template=template,
                llm_s=latency,
                prediction=prediction,
                scores=scores,
                error=error,
                repeat_latencies=repeats,
            )
        )
        tag = "OK " if prediction is not None else "ERR"
        sys.stdout.write(
            f"  [{tag}] T{template} {bucket:<15} {doc_id}  llm={latency:.1f}s\n"
        )
        sys.stdout.flush()

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def _bucket_by_split(results: list[DocResult]) -> dict[str, list[DocResult]]:
    buckets: dict[str, list[DocResult]] = {"clean": [], "scanned-light": [], "scanned-heavy": []}
    for r in results:
        if r.split in buckets:
            buckets[r.split].append(r)
    return buckets


def _bucket_by_template(results: list[DocResult]) -> dict[int, list[DocResult]]:
    buckets: dict[int, list[DocResult]] = {1: [], 2: [], 3: []}
    for r in results:
        buckets[r.template].append(r)
    return buckets


def _summarize_bucket(results: list[DocResult]) -> dict:
    """Aggregate scores for one bucket.

    For each field returns mean score AND the number of documents with a
    perfect score (score == 1.0). This gives, e.g., ``0.93 (14/15)`` cells.
    """
    if not results:
        return {}
    per_field_strict: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    per_field_norm: dict[str, list[float]] = {f: [] for f in ALL_FIELDS}
    latencies: list[float] = []
    invalid = 0
    for r in results:
        if r.prediction is None:
            invalid += 1
        if r.llm_s is not None:
            latencies.append(r.llm_s)
        for f in ALL_FIELDS:
            per_field_strict[f].append(r.scores[f].strict)
            per_field_norm[f].append(r.scores[f].normalized)
    n = len(results)
    return {
        "n_docs": n,
        "invalid_pct": 100.0 * invalid / n,
        "latency_median_s": statistics.median(latencies) if latencies else 0.0,
        "latency_p90_s": _percentile(latencies, 90),
        "per_field": {
            f: {
                "strict_mean": _mean(per_field_strict[f]),
                "strict_perfect": sum(1 for x in per_field_strict[f] if x == 1.0),
                "normalized_mean": _mean(per_field_norm[f]),
                "normalized_perfect": sum(1 for x in per_field_norm[f] if x == 1.0),
                "n": n,
            }
            for f in ALL_FIELDS
        },
    }


def _cell(v: dict, key: str) -> str:
    """Format one score cell as ``mean (perfect/n)``."""
    return f"{v[f'{key}_mean']:.2f} ({v[f'{key}_perfect']}/{v['n']})"


def _render_section(title: str, summary: dict | None) -> list[str]:
    lines: list[str] = [f"### {title}", ""]
    if not summary:
        lines.append("_No document in this bucket for this run._")
        lines.append("")
        return lines
    lines.append(
        f"- n = {summary['n_docs']}, invalid JSON = {summary['invalid_pct']:.1f}%, "
        f"latency median = {summary['latency_median_s']:.1f}s, "
        f"p90 = {summary['latency_p90_s']:.1f}s"
    )
    lines.append("")
    lines.append("| Field | Strict | Normalized |")
    lines.append("|---|---:|---:|")
    for f in ALL_FIELDS:
        v = summary["per_field"][f]
        lines.append(f"| {f} | {_cell(v, 'strict')} | {_cell(v, 'normalized')} |")
    lines.append("")
    return lines


def _render_markdown(
    model: str, started_at: str, results: list[DocResult], split_dev: bool
) -> str:
    by_split = _bucket_by_split(results)
    by_template = _bucket_by_template(results)
    split_summaries = {name: _summarize_bucket(rs) for name, rs in by_split.items()}
    template_summaries = {t: _summarize_bucket(rs) for t, rs in by_template.items()}

    lines: list[str] = []
    lines.append(f"# Evaluation report — `{model}`")
    lines.append("")
    lines.append(f"- Started at: {started_at}")
    lines.append(f"- Documents evaluated: {len(results)}")
    if split_dev:
        lines.append(
            f"- `--split-dev` on: {len(HELDOUT_DEV_IDS)} dev documents held out "
            f"({', '.join(sorted(HELDOUT_DEV_IDS))})"
        )
    lines.append(
        "- Each cell shows ``mean (docs_with_perfect_score / total)``."
    )
    lines.append("")

    lines.append("## Scores by degradation split")
    lines.append("")
    for bucket_name in ("clean", "scanned-light", "scanned-heavy"):
        lines.extend(_render_section(bucket_name, split_summaries.get(bucket_name) or {}))

    lines.append("## Scores by template")
    lines.append("")
    for t in (1, 2, 3):
        lines.extend(_render_section(f"T{t}", template_summaries.get(t) or {}))

    # Variance subsection (only if any repeats were collected).
    rep = [(r.doc_id, r.split, [r.llm_s] + r.repeat_latencies) for r in results if r.repeat_latencies]
    if rep:
        lines.append("## Latency variance (per document with --repeat)")
        lines.append("")
        lines.append("| Doc | Split | Runs | Median | Stdev |")
        lines.append("|---|---|---:|---:|---:|")
        for doc_id, split, lats in rep:
            lines.append(
                f"| {doc_id} | {split} | {len(lats)} | "
                f"{statistics.median(lats):.1f} | {statistics.pstdev(lats):.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _write_outputs(
    model: str, results: list[DocResult], split_dev: bool
) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    jsonl_path = RESULTS_DIR / f"{safe_model}_{stamp}.jsonl"
    md_path = REPORTS_DIR / f"{safe_model}_{stamp}.md"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "doc_id": r.doc_id,
                        "split": r.split,
                        "template": r.template,
                        "llm_s": r.llm_s,
                        "repeat_latencies_s": r.repeat_latencies,
                        "error": r.error,
                        "prediction": r.prediction,
                        "scores": {
                            k: {"strict": v.strict, "normalized": v.normalized}
                            for k, v in r.scores.items()
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    md_path.write_text(
        _render_markdown(model, stamp, results, split_dev), encoding="utf-8"
    )
    return jsonl_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Ollama tag (sets OLLAMA_MODEL).")
    p.add_argument(
        "--split",
        choices=["clean", "scanned", "all"],
        default="scanned",
        help="Which PDF split to evaluate.",
    )
    p.add_argument("--limit", type=int, default=None, help="First N docs only.")
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each LLM call K times for latency-variance measurement.",
    )
    p.add_argument(
        "--split-dev",
        action="store_true",
        help=(
            "Exclude the 10 fixed held-out docs (HELDOUT_DEV_IDS) from the run "
            "so published scores are computed on the remaining 20."
        ),
    )
    p.add_argument(
        "--only-dev",
        action="store_true",
        help="Keep only the 10 fixed held-out docs (for prompt tuning).",
    )
    args = p.parse_args(argv)
    if args.split_dev and args.only_dev:
        p.error("--split-dev and --only-dev are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    print(
        f"Model={args.model} split={args.split} limit={args.limit} "
        f"repeat={args.repeat} split_dev={args.split_dev}"
    )
    results = asyncio.run(evaluate(args))
    jsonl_path, md_path = _write_outputs(args.model, results, args.split_dev)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
