#!/usr/bin/env python3
"""Ingest all 10 LoCoMo conversations into fresh stores using the FIXED fastembed
binary (real semantic embeddings). Uses HMG_SERVER_BIN env to point at fixed binary."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import hmg_client as lib

DOMAIN = "software-engineering"
CTX = {"tenant_id": "tenant-acme", "workspace": "platform", "repository": "locomo-benchmark"}

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("benchmark_stores_fe_all")
    out.mkdir(parents=True, exist_ok=True)
    ds = lib.load_dataset(Path("locomo10.json"))
    import os
    print(f"HMG_SERVER_BIN = {os.environ.get('HMG_SERVER_BIN', '(default)')}", flush=True)
    t_all = time.time()
    for i, s in enumerate(ds):
        sid = s["sample_id"]; store = out / sid
        if (store / "partitions").exists():
            print(f"[{i}] {sid} already ingested, skip", flush=True); continue
        ctx = dict(CTX); ctx["branch"] = sid
        turns = lib.collect_turns(s["conversation"])
        t0 = time.time()
        with lib.McpClient(store) as c:
            for sk, st, t in turns:
                c.call_tool("memory_memorize", {"content": lib.build_memory_text(sk, st, t), "context": ctx,
                    "domain_pack_id": DOMAIN, "modality": "dialogue", "source": "locomo-benchmark",
                    "disable_redaction": True, "disable_admission": True})
        print(f"[{i}] {sid} ingested {len(turns)} turns in {time.time()-t0:.0f}s", flush=True)
    print(f"ALL INGEST DONE in {time.time()-t_all:.0f}s", flush=True)

if __name__ == "__main__":
    main()
