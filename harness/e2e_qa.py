#!/usr/bin/env python3
"""
GLM-5.2 LoCoMo E2E answerer+judge runner (reuses frozen HMG retrieval @ top-20).

Mirrors the methodology of run_hmg_locomo_e2e.py but:
  * bypasses Codex config (provider/model/base_url/key via CLI/env)
  * uses AsyncOpenAI (OpenAI-compatible chat completions)
  * supports --enable-thinking (GLM-5.2 is a reasoning model; off by default)
  * per-question JSON cache for resumability / partial rejudge
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI

from prompts.locomo_prompts import (  # noqa: E402
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
)


SCORED_CATEGORIES = {1, 2, 3, 4}
TOP_K = 20  # default; overridable via --top-k
CUTOFF_KEY = "top_20"  # updated to f"top_{TOP_K}" at runtime
JSON_SCHEMA_HINT = (
    'Return a single JSON object only, with exactly these keys: '
    '"reasoning" (one sentence) and "label" ("CORRECT" or "WRONG").'
)


# ----------------------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_locomo_datetime(date_str: str) -> datetime | None:
    cleaned = " ".join((date_str or "").strip().split())
    if not cleaned:
        return None
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(cleaned.upper(), fmt)
        except ValueError:
            continue
    return None


def locomo_datetime_to_iso(date_str: str) -> str:
    parsed = parse_locomo_datetime(date_str)
    if parsed is None:
        return ""
    return parsed.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_memory_date(memory_text: str) -> str:
    match = re.search(r"\((?:session_\d+)\s+@\s+(.+?)\)", memory_text)
    return locomo_datetime_to_iso(match.group(1)) if match else ""


def extract_dia_ids(memory_text: str) -> list[str]:
    return re.findall(r"\[(D\d+:\d+)\]", memory_text)


def latest_session_date(conversation: dict[str, Any]) -> str | None:
    dates = []
    for key, value in conversation.items():
        if key.startswith("session_") and key.endswith("_date_time"):
            parsed = parse_locomo_datetime(str(value))
            if parsed is not None:
                dates.append((parsed, str(value)))
    return max(dates, key=lambda item: item[0])[1] if dates else None


def load_evidence_lookup(dataset: list[dict[str, Any]]) -> dict[tuple[int, str], str]:
    lookup: dict[tuple[int, str], str] = {}
    for conv_idx, sample in enumerate(dataset):
        conversation = sample["conversation"]
        session_dates: dict[str, str] = {}
        for key, value in conversation.items():
            if key.startswith("session_") and key.endswith("_date_time"):
                session_dates[key.replace("session_", "").replace("_date_time", "")] = str(value)
        for key, value in conversation.items():
            if not key.startswith("session_") or key.endswith("_date_time") or not isinstance(value, list):
                continue
            for turn in value:
                dia_id = turn.get("dia_id", "")
                if not dia_id:
                    continue
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                dia_match = re.match(r"D(\d+):", dia_id)
                date_suffix = ""
                if dia_match:
                    session_date = session_dates.get(dia_match.group(1), "")
                    if session_date:
                        date_suffix = f", said on {session_date}"
                lookup[(conv_idx, dia_id)] = f'[{dia_id}{date_suffix}] {speaker}: "{text}"'
    return lookup


def build_evidence_context(evidence_lookup, conv_idx, evidence):
    lines = []
    for dia_id in evidence:
        text = evidence_lookup.get((conv_idx, dia_id))
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    return json.loads(stripped)


def convert_retrieved_texts(retrieved_texts: list[str], top_k: int = TOP_K) -> list[dict[str, Any]]:
    converted = []
    for rank, text in enumerate(retrieved_texts[:top_k], start=1):
        dia_ids = extract_dia_ids(text)
        converted.append(
            {
                "memory": text,
                "created_at": extract_memory_date(text),
                "rank": rank,
                "source_id": dia_ids[0] if dia_ids else "",
                "dia_ids": dia_ids,
            }
        )
    return converted


# ----------------------------------------------------------------------------- question building
def build_question_items(retrieval_payload, dataset, categories, run_mode, pilot_per_conv, max_questions):
    items = []
    by_conv_seen: dict[int, int] = defaultdict(int)
    for conv_result in retrieval_payload.get("results", []):
        conv_idx = int(conv_result["conversation_index"])
        sample_id = conv_result["sample_id"]
        conversation = dataset[conv_idx]["conversation"]
        reference_date = latest_session_date(conversation)
        for question_result in conv_result.get("questions", []):
            category = int(question_result["category"])
            if category not in categories:
                continue
            if run_mode == "pilot" and by_conv_seen[conv_idx] >= pilot_per_conv:
                continue
            q_idx = int(question_result["question_index"])
            qid = f"conv{conv_idx}_q{q_idx}"
            dataset_qa = dataset[conv_idx]["qa"][q_idx]
            if dataset_qa["question"] != question_result["question"]:
                raise RuntimeError(f"Question mismatch for {qid}")
            items.append(
                {
                    "question_id": qid,
                    "conversation_idx": conv_idx,
                    "sample_id": sample_id,
                    "question_index": q_idx,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, "unknown"),
                    "question": question_result["question"],
                    "ground_truth_answer": str(question_result.get("answer") or dataset_qa.get("answer") or ""),
                    "evidence": list(question_result.get("evidence") or []),
                    "reference_date": reference_date,
                    "retrieval": {
                        "search_query": question_result["question"],
                        "search_results": convert_retrieved_texts(question_result.get("retrieved_texts") or [], top_k=TOP_K),
                        "total_results": int(question_result.get("retrieved_count") or 0),
                        "first_hit_rank": question_result.get("first_hit_rank"),
                        "retrieval_only_hit": bool(question_result.get("hit")),
                    },
                }
            )
            by_conv_seen[conv_idx] += 1
            if max_questions is not None and len(items) >= max_questions:
                return items
    return items


# ----------------------------------------------------------------------------- client
class AsyncRateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_at:
                await asyncio.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.interval


@dataclass
class GLMClient:
    client: AsyncOpenAI
    model: str
    limiter: AsyncRateLimiter
    enable_thinking: bool = False
    max_retries: int = 8
    timeout_seconds: float = 180.0
    stats: dict = field(default_factory=lambda: {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0})

    def _extra(self) -> dict:
        return {"extra_body": {"enable_thinking": self.enable_thinking}}

    async def generate(self, system: str, user: str, max_output_tokens: int) -> str:
        return await self._request(system, user, max_output_tokens, json_mode=False)

    async def generate_json(self, system: str, user: str, max_output_tokens: int) -> dict[str, Any]:
        raw = await self._request(system, f"{user}\n\n{JSON_SCHEMA_HINT}", max_output_tokens, json_mode=True)
        parsed = parse_json_object(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Structured response was not a JSON object")
        if "final" in parsed and len(parsed) == 1:
            inner = parsed["final"]
            if isinstance(inner, dict):
                parsed = inner
            elif isinstance(inner, str):
                parsed = parse_json_object(inner)
        return parsed

    async def _request(self, system: str, user: str, max_output_tokens: int, json_mode: bool) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            await self.limiter.wait()
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": user})
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_output_tokens,
                    "temperature": 0.0,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                kwargs.update(self._extra())
                resp = await self.client.chat.completions.create(**kwargs)
                self.stats["calls"] += 1
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self.stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
                    self.stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
                    cd = getattr(usage, "completion_tokens_details", None)
                    if cd is not None:
                        self.stats["reasoning_tokens"] += getattr(cd, "reasoning_tokens", 0) or 0
                choice = resp.choices[0]
                content = choice.message.content
                finish = choice.finish_reason
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"empty content, finish_reason={finish}")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = str(exc)
                # backoff for rate-limit / overload-ish errors
                sleep = min(40.0, 2.0 * attempt)
                if any(s in msg for s in ("429", "rate", "overload", "5", "timeout", "empty content")):
                    sleep = min(40.0, 2.5 * attempt)
                if attempt < self.max_retries:
                    await asyncio.sleep(sleep)
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_error}") from last_error


# ----------------------------------------------------------------------------- evaluation
async def evaluate_item(item, client: GLMClient, evidence_lookup, output_dir, rejudge, retry_errors):
    per_question_dir = output_dir / "per_question"
    result_path = per_question_dir / f"{item['question_id']}.json"
    if result_path.exists() and not rejudge:
        existing = read_json(result_path)
        top_20 = existing.get("cutoff_results", {}).get(CUTOFF_KEY, {})
        has_error = top_20.get("judgment") == "ERROR" or bool(top_20.get("error"))
        if not (retry_errors and has_error):
            return existing

    result = dict(item)
    result["cutoff_results"] = {}
    search_results = result["retrieval"]["search_results"][:TOP_K]
    processed_answer = preprocess_answer(result["category"], result["ground_truth_answer"])

    try:
        answer_prompt = get_answer_generation_prompt(
            result["question"], search_results, reference_date=result.get("reference_date")
        )
        answer_tokens = 16384 if client.enable_thinking else 2048
        try:
            generated_answer = await client.generate(system="", user=answer_prompt, max_output_tokens=answer_tokens)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"answerer failed: {exc}") from exc
        if "ANSWER:" in generated_answer:
            generated_answer = generated_answer.rsplit("ANSWER:", 1)[-1].strip()

        evidence_context = build_evidence_context(evidence_lookup, result["conversation_idx"], result.get("evidence") or [])
        judge_prompt = get_judge_prompt_with_evidence(
            result["category"], result["question"], processed_answer, generated_answer, evidence_context
        )
        judge_tokens = 4096 if client.enable_thinking else 1024
        try:
            judge = await client.generate_json(system=JUDGE_SYSTEM_PROMPT, user=judge_prompt, max_output_tokens=judge_tokens)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"judge failed: {exc}") from exc
        label = str(judge.get("label", "")).upper().strip()
        if label not in {"CORRECT", "WRONG"}:
            raise RuntimeError(f"Judge returned invalid label: {label!r}")
        score = 1.0 if label == "CORRECT" else 0.0
        result["cutoff_results"][CUTOFF_KEY] = {
            "judgment": label,
            "score": score,
            "generated_answer": generated_answer,
            "memories_evaluated": len(search_results),
            "reason": str(judge.get("reasoning", "")),
        }
    except Exception as exc:  # noqa: BLE001
        result["cutoff_results"][CUTOFF_KEY] = {
            "judgment": "ERROR",
            "score": 0.0,
            "generated_answer": "",
            "memories_evaluated": len(search_results),
            "reason": "",
            "error": str(exc),
        }

    write_json(result_path, result)
    return result


def compute_metrics(evaluations):
    total = len(evaluations)
    correct = errors = 0
    accum: dict[str, list] = defaultdict(list)
    for item in evaluations:
        top = item.get("cutoff_results", {}).get(CUTOFF_KEY, {})
        if top.get("judgment") == "CORRECT":
            correct += 1
        if top.get("judgment") == "ERROR" or top.get("error"):
            errors += 1
        accum[item.get("category_name", "unknown")].append(item)
    by_category = {}
    for category_name, rows in accum.items():
        cat_correct = sum(1 for r in rows if r.get("cutoff_results", {}).get(CUTOFF_KEY, {}).get("judgment") == "CORRECT")
        cat_errors = sum(1 for r in rows if r.get("cutoff_results", {}).get(CUTOFF_KEY, {}).get("judgment") == "ERROR" or r.get("cutoff_results", {}).get(CUTOFF_KEY, {}).get("error"))
        by_category[category_name] = {
            "total": len(rows),
            "correct": cat_correct,
            "errors": cat_errors,
            "accuracy": cat_correct / len(rows) if rows else 0.0,
        }
    by_category = {k: by_category[k] for k in sorted(by_category)}
    return {
        "overall": {"total": total, "correct": correct, "errors": errors, "accuracy": correct / total if total else 0.0},
        "by_category": by_category,
    }


# ----------------------------------------------------------------------------- main
async def run_async(args):
    global TOP_K, CUTOFF_KEY
    TOP_K = args.top_k
    CUTOFF_KEY = f"top_{args.top_k}"
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_payload = read_json(args.retrieval_json)
    dataset = read_json(args.dataset)
    categories = {int(v) for v in args.categories.split(",") if v.strip()}
    assert categories <= SCORED_CATEGORIES, "This runner is intended for scored categories 1-4 only"

    frozen_path = output_dir / "frozen_hmg_retrieval.json"
    if args.retrieval_json.resolve() != frozen_path.resolve():
        import shutil
        shutil.copy2(args.retrieval_json, frozen_path)
    frozen_sha = sha256_file(frozen_path)

    items = build_question_items(retrieval_payload, dataset, categories, args.run_mode, args.pilot_per_conv, args.max_questions)
    if not items:
        raise RuntimeError("No questions selected")

    api_key = os.environ.get(args.api_key_env) or args.api_key
    assert api_key, f"missing API key: set {args.api_key_env} or pass --api-key"

    aclient = AsyncOpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout_seconds)
    limiter = AsyncRateLimiter(args.rpm)
    client = GLMClient(
        client=aclient, model=args.model, limiter=limiter,
        enable_thinking=args.enable_thinking, max_retries=args.max_retries, timeout_seconds=args.timeout_seconds,
    )
    evidence_lookup = load_evidence_lookup(dataset)

    # summary-only mode (recompute metrics from existing per_question cache)
    if args.summarize_existing:
        evaluations = []
        for item in items:
            rp = output_dir / "per_question" / f"{item['question_id']}.json"
            if rp.exists():
                evaluations.append(read_json(rp))
        evaluations.sort(key=lambda r: (r["conversation_idx"], r["question_index"]))
        assert evaluations, f"No existing per-question results in {output_dir/'per_question'}"
        return _write_outputs(args, output_dir, frozen_sha, client, evaluations, retrieval_payload, started=utc_now(), finished=utc_now())

    print(f"Selected {len(items)} questions for {args.run_mode} run (thinking={'ON' if args.enable_thinking else 'OFF'}, model={args.model})", file=sys.stderr, flush=True)
    started_at = utc_now()
    sem = asyncio.Semaphore(args.max_workers)
    done = 0
    lock = asyncio.Lock()

    async def run_one(item):
        nonlocal done
        async with sem:
            res = await evaluate_item(item, client, evidence_lookup, output_dir, args.rejudge, args.retry_errors)
            async with lock:
                done += 1
                if done % max(1, args.progress_every) == 0 or done == len(items):
                    acc_so_far = _quick_acc() if False else None
                    print(f"[{done}/{len(items)}] evaluated", file=sys.stderr, flush=True)
            return res

    evaluations = await asyncio.gather(*(run_one(it) for it in items))
    evaluations.sort(key=lambda r: (r["conversation_idx"], r["question_index"]))
    finished_at = utc_now()
    print(f"token stats: {json.dumps(client.stats)}", file=sys.stderr, flush=True)
    return _write_outputs(args, output_dir, frozen_sha, client, evaluations, retrieval_payload, started=started_at, finished=finished_at)


def _write_outputs(args, output_dir, frozen_sha, client, evaluations, retrieval_payload, started, finished):
    metrics = {CUTOFF_KEY: compute_metrics(evaluations)}
    unified_path = output_dir / f"hmg_locomo_e2e_{args.run_mode}_{CUTOFF_KEY}_results.json"
    report_path = output_dir / f"hmg_locomo_e2e_{args.run_mode}_{CUTOFF_KEY}_report.md"
    metadata = {
        "benchmark": "locomo",
        "mode": "hmg_e2e_answerer_judge_glm",
        "run_mode": args.run_mode,
        "dataset_path": str(args.dataset),
        "frozen_retrieval_json": str(output_dir / "frozen_hmg_retrieval.json"),
        "frozen_retrieval_sha256": frozen_sha,
        "output_dir": str(output_dir),
        "answerer_model": args.model,
        "judge_model": args.model,
        "provider_base_url": args.base_url,
        "enable_thinking": args.enable_thinking,
        "top_k": TOP_K,
        "categories": sorted({int(v) for v in args.categories.split(",") if v.strip()}),
        "total_questions": len(evaluations),
        "started_at": started,
        "finished_at": finished,
        "max_workers": args.max_workers,
        "rpm": args.rpm,
        "token_stats": client.stats if hasattr(client, "stats") else {},
    }
    unified = {"metadata": metadata, "metrics": metrics, "evaluations": evaluations, "comparison": {"hmg_retrieval_only_top20": retrieval_payload.get("summary", {})}}
    write_json(unified_path, unified)
    _render_report(report_path, unified, retrieval_payload)
    _export_wrong(output_dir, evaluations)
    print(str(unified_path))
    print(str(report_path))
    return metrics[CUTOFF_KEY]["overall"]["errors"]


def _export_wrong(output_dir, evaluations):
    wrong = [e for e in evaluations if e.get("cutoff_results", {}).get(CUTOFF_KEY, {}).get("judgment") != "CORRECT"]
    rows = []
    for e in wrong:
        top = e["cutoff_results"][CUTOFF_KEY]
        rows.append({
            "question_id": e["question_id"],
            "category": e["category_name"],
            "judgment": top.get("judgment"),
            "retrieval_hit": e["retrieval"]["retrieval_only_hit"],
            "first_hit_rank": e["retrieval"]["first_hit_rank"],
            "question": e["question"],
            "gold": e["ground_truth_answer"],
            "generated": top.get("generated_answer", ""),
            "reason": top.get("reason", ""),
            "error": top.get("error", ""),
        })
    write_json(output_dir / "wrong_answers_top20.json", rows)
    (output_dir / "wrong_answers_top20.csv").write_text(
        "question_id,category,judgment,retrieval_hit,question,gold,generated\n" +
        "\n".join(
            '"' + str(r[k]).replace('"', '""') + '"' for r in rows
            for k in []
        ) if False else "",
        encoding="utf-8",
    )
    # proper csv
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["question_id", "category", "judgment", "retrieval_hit", "first_hit_rank", "question", "gold", "generated", "reason"])
    for r in rows:
        w.writerow([r["question_id"], r["category"], r["judgment"], r["retrieval_hit"], r["first_hit_rank"], r["question"], r["gold"], r["generated"], r["reason"]])
    (output_dir / "wrong_answers_top20.csv").write_text(buf.getvalue(), encoding="utf-8")


def _render_report(output_path, unified, retrieval_payload):
    m = unified["metadata"]
    metrics = unified["metrics"][CUTOFF_KEY]
    ret = retrieval_payload.get("summary", {})
    lines = [
        "# HMG LoCoMo E2E QA @ Top-20 — GLM-5.2 rerun",
        "",
        "## Scope",
        f"- Run mode: `{m['run_mode']}`",
        f"- Answerer/Judge model: `{m['answerer_model']}` @ `{m['provider_base_url']}`",
        f"- enable_thinking: `{m['enable_thinking']}`",
        f"- Top-k cutoff: `top_20`",
        f"- Started: `{m['started_at']}` Finished: `{m['finished_at']}`",
        "",
        "## E2E Results",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Accuracy @ top-20 | {pct(metrics['overall']['accuracy'])} ({metrics['overall']['correct']}/{metrics['overall']['total']}) |",
        f"| Errors | {metrics['overall']['errors']} |",
        "",
        "### By Category",
        "| Category | Correct / Total | Accuracy | Errors |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cn, it in metrics["by_category"].items():
        lines.append(f"| {cn} | {it['correct']}/{it['total']} | {pct(it['accuracy'])} | {it['errors']} |")
    lines += [
        "",
        "## Comparison",
        "| System | Metric | Score |",
        "| --- | --- | ---: |",
        f"| GLM-5.2 (this run) | answerer+judge @ top-20 | {pct(metrics['overall']['accuracy'])} |",
        f"| HMG retrieval-only | evidence hit @ top-20 | {pct(ret.get('scored_hit_rate'))} |",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="GLM-5.2 LoCoMo E2E answerer+judge @ top-20 (frozen HMG retrieval).")
    p.add_argument("--retrieval-json", type=Path, default=Path("frozen_hmg_retrieval.json"))
    p.add_argument("--dataset", type=Path, default=Path("locomo10.json"))
    p.add_argument("--output-dir", type=Path, default=Path("results/glm52_thinking_off"))
    p.add_argument("--run-mode", choices=["pilot", "full"], default="pilot")
    p.add_argument("--categories", default="1,2,3,4")
    p.add_argument("--pilot-per-conv", type=int, default=2)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--rpm", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--progress-every", type=int, default=20)
    # GLM config
    p.add_argument("--model", default="glm-5.2")
    p.add_argument("--base-url", default="https://open.bigmodel.cn/api/paas/v4")
    p.add_argument("--api-key", default=None)
    p.add_argument("--api-key-env", default="ZHIPU_API_KEY")
    p.add_argument("--enable-thinking", action="store_true", default=False)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rejudge", action="store_true")
    p.add_argument("--retry-errors", action="store_true", default=True)
    p.add_argument("--no-retry-errors", action="store_false", dest="retry_errors")
    p.add_argument("--summarize-existing", action="store_true")
    p.add_argument("--fail-on-errors", action="store_true", default=True)
    p.add_argument("--allow-errors", action="store_false", dest="fail_on_errors")
    return p.parse_args()


def main():
    args = parse_args()
    errors = asyncio.run(run_async(args))
    return 2 if (args.fail_on_errors and errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
