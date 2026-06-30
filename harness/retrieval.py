#!/usr/bin/env python3
"""Full HMG retrieval on all 10 conversations with adaptive top-50 config.
Produces E2E-compatible frozen retrieval JSON (retrieved_texts = top-50)."""
from __future__ import annotations
import argparse, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import hmg_client as lib

DOMAIN_PACK = "software-engineering"
CATEGORY_NAMES = lib.CATEGORY_NAMES
SCORED = lib.SCORED_CATEGORIES
CTX = {"tenant_id": "tenant-acme", "workspace": "platform", "repository": "locomo-benchmark"}

RECALL_CFG = dict(
    max_results=50, precision_mode=False, disable_adaptive_recall=False,
    disable_embedding_noise_gate=True, disable_seed_length_penalty=False, disable_lifecycle_ranking=False,
)


def parse_ids(text):
    return lib.parse_dia_ids(text)


def eval_question(retrieved_texts, evidence):
    ids = []
    for t in retrieved_texts:
        ids.extend(parse_ids(t))
    pos = [i for i, d in enumerate(ids, 1) if d in evidence]
    return bool(pos), (min(pos) if pos else None), ids


def run_conv(i, sample, stores_dir, ctx_base):
    sid = sample["sample_id"]
    store = stores_dir / sid
    ctx = dict(ctx_base); ctx["branch"] = sid
    qres = []
    with lib.McpClient(store) as c:
        for q_idx, qa in enumerate(sample["qa"]):
            ra = {"query": qa["question"], "context": ctx, "domain_pack_id": DOMAIN_PACK}
            ra.update(RECALL_CFG)
            data = c.call_tool("memory_recall", ra)
            texts = [a.get("text", "") for a in data.get("atoms", [])]
            ev = qa.get("evidence") or []
            hit, fhr, ids = eval_question(texts, ev)
            # hit@20 and hit@50
            ids20 = []
            for t in texts[:20]:
                ids20.extend(parse_ids(t))
            hit20 = any(d in ev for d in ids20)
            qres.append({
                "question_index": q_idx, "question": qa["question"], "category": qa["category"],
                "category_name": CATEGORY_NAMES.get(qa["category"], "unknown"),
                "answer": qa.get("answer") or qa.get("adversarial_answer"),
                "evidence": ev, "retrieved_count": len(texts),
                "retrieved_texts": texts, "hit": hit20, "hit_at_50": hit,
                "first_hit_rank": fhr, "retrieved_ids": ids,
            })
            if (q_idx + 1) % 40 == 0:
                h20 = sum(1 for q in qres if q["hit"]) / len(qres)
                h50 = sum(1 for q in qres if q["hit_at_50"]) / len(qres)
                print(f"  [{sid}] q{q_idx+1}/{len(sample['qa'])} hit@20={h20*100:.1f}% hit@50={h50*100:.1f}%", file=sys.stderr, flush=True)
    return {"sample_id": sid, "conversation_index": i, "questions": qres}


def aggregate(results):
    st_total = st_h20 = st_h50 = 0
    by_cat = defaultdict(lambda: {"total": 0, "h20": 0, "h50": 0})
    ranks = []
    for conv in results:
        for q in conv["questions"]:
            cat = q["category"]
            by_cat[cat]["total"] += 1
            by_cat[cat]["h20"] += int(q["hit"])
            by_cat[cat]["h50"] += int(q["hit_at_50"])
            if q["first_hit_rank"] is not None:
                ranks.append(q["first_hit_rank"])
            if cat in SCORED:
                st_total += 1
                st_h20 += int(q["hit"])
                st_h50 += int(q["hit_at_50"])
    cat_metrics = {}
    for cat, c in sorted(by_cat.items()):
        cat_metrics[str(cat)] = {"name": CATEGORY_NAMES.get(cat, "?"), "total": c["total"],
                                  "hits_top20": c["h20"], "hit_rate_top20": c["h20"]/c["total"] if c["total"] else 0,
                                  "hits_top50": c["h50"], "hit_rate_top50": c["h50"]/c["total"] if c["total"] else 0}
    return {"scored_total": st_total, "scored_hits_top20": st_h20, "scored_hit_rate_top20": st_h20/st_total if st_total else 0,
            "scored_hits_top50": st_h50, "scored_hit_rate_top50": st_h50/st_total if st_total else 0,
            "category_metrics": cat_metrics, "first_hit_rank_avg": (sum(ranks)/len(ranks)) if ranks else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("locomo10.json"))
    ap.add_argument("--stores-dir", type=Path, default=Path("benchmark_stores_all"))
    ap.add_argument("--output", type=Path, default=Path("frozen_retrieval_top50.json"))
    ap.add_argument("--limit-conversations", type=int, default=None)
    args = ap.parse_args()

    ds = lib.load_dataset(args.dataset)
    if args.limit_conversations:
        ds = ds[: args.limit_conversations]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results = []
    for i, s in enumerate(ds):
        print(f"[{i+1}/{len(ds)}] {s['sample_id']}...", file=sys.stderr, flush=True)
        results.append(run_conv(i, s, args.stores_dir, CTX))
    summary = aggregate(results)
    payload = {"benchmark": "locomo", "mode": "retrieval_adaptive_top50", "top_k": 50,
               "recall_config": RECALL_CFG, "started_at": started,
               "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "dataset_path": str(args.dataset), "stores_dir": str(args.stores_dir),
               "results": results, "summary": summary}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== SUMMARY adaptive top-50 ===", file=sys.stderr)
    print(f"scored hit@20: {summary['scored_hits_top20']}/{summary['scored_total']} = {summary['scored_hit_rate_top20']*100:.2f}%", file=sys.stderr)
    print(f"scored hit@50: {summary['scored_hits_top50']}/{summary['scored_total']} = {summary['scored_hit_rate_top50']*100:.2f}%", file=sys.stderr)
    for cat, m in summary["category_metrics"].items():
        if int(cat) in SCORED:
            print(f"  {m['name']}: hit@20={m['hit_rate_top20']*100:.1f}%  hit@50={m['hit_rate_top50']*100:.1f}%  ({m['total']})", file=sys.stderr)
    print(str(args.output), file=sys.stderr)


if __name__ == "__main__":
    main()
