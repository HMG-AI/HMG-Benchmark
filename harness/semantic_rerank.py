#!/usr/bin/env python3
"""Semantic rerank with LOCAL e5-small ONNX (no download). Rerank fusion pool by real
semantic embeddings, measure hit@k vs hash order. Validates semantic embedding value."""
import json, re, sys, time, os, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import tokenizers
import onnxruntime as ort

def _find_e5_snapshot():
    """Locate the local multilingual-e5-small snapshot dir (HMG embedding cache or HF hub)."""
    env = os.environ.get("E5_SNAPSHOT")
    if env and Path(env).is_dir():
        return env
    candidates = [
        Path.home() / ".local/share/hmg/embedding-cache/models--intfloat--multilingual-e5-small",
        Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small",
    ]
    for base in candidates:
        snaps = sorted(glob.glob(str(base / "snapshots" / "*")))
        if snaps:
            return snaps[0]
    raise SystemExit(
        "multilingual-e5-small snapshot not found. Set E5_SNAPSHOT=<dir> or run "
        "`hmg model embedding download` / `huggingface-cli download intfloat/multilingual-e5-small`."
    )

SNAP = _find_e5_snapshot()

def parse_ids(t): return re.findall(r"\[(D\d+:\d+)\]", t)
def ids_set(texts):
    s=set()
    for t in texts:
        for d in parse_ids(t): s.add(d)
    return s
def hit_at(texts,ev,kk): return bool(ids_set(texts[:kk]) & set(ev))
SCORED={1,2,3,4}

def make_embedder():
    tok=tokenizers.Tokenizer.from_file(f"{SNAP}/tokenizer.json")
    tok.enable_padding(length=512)
    sess=ort.InferenceSession(f"{SNAP}/onnx/model.onnx", providers=["CPUExecutionProvider"])
    inputs={i.name:i for i in sess.get_inputs()}
    inp=inputs.get("input_ids"); attn_name=inputs.get("attention_mask"); tt_name=inputs.get("token_type_ids")
    def embed(texts, prefix, batch=64):
        all_out=[]
        for b in range(0,len(texts),batch):
            chunk=texts[b:b+batch]
            enc=tok.encode_batch([prefix+t for t in chunk])
            ids=np.array([e.ids for e in enc], dtype=np.int64)
            am=np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds={"input_ids":ids, "attention_mask":am}
            if tt_name: feeds["token_type_ids"]=np.zeros_like(ids)
            out=sess.run(None, feeds)[0]
            mask=am[...,None].astype(np.float32)
            pooled=(out*mask).sum(1)/np.clip(mask.sum(1),1e-9,None)
            pooled=pooled/np.clip(np.linalg.norm(pooled,axis=1,keepdims=True),1e-9,None)
            all_out.append(pooled)
        return np.vstack(all_out)
    return embed

def main():
    d=json.loads(Path("fused_dual_top150.json").read_text())
    all_texts=set(); qs=[]
    for c in d["results"]:
        for q in c["questions"]:
            if q["category"] in SCORED:
                qs.append(q)
                for t in q.get("retrieved_texts") or []: all_texts.add(t)
    all_texts=list(all_texts)
    print(f"questions={len(qs)} pool_atoms={len(all_texts)}", file=sys.stderr, flush=True)
    emb=make_embedder()
    print("embedding atoms (passage:)...", file=sys.stderr, flush=True)
    t0=time.time()
    atom_emb=emb(all_texts,"passage: ")
    print(f"  {atom_emb.shape} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    queries=[q["question"] for q in qs]
    print("embedding queries (query:)...", file=sys.stderr, flush=True)
    t0=time.time()
    q_emb=emb(queries,"query: ")
    print(f"  {q_emb.shape} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    atom_idx={t:i for i,t in enumerate(all_texts)}
    cutoffs=[10,20,50,100,150]
    hh={c:[0,0] for c in cutoffs}; sh={c:[0,0] for c in cutoffs}
    for i,q in enumerate(qs):
        pool=q.get("retrieved_texts") or []; ev=q.get("evidence") or []
        for c in cutoffs:
            hh[c][0]+=int(hit_at(pool,ev,c)); hh[c][1]+=1
        if pool:
            pi=[atom_idx[t] for t in pool if t in atom_idx]
            sims=atom_emb[pi] @ q_emb[i]
            reranked=[pool[j] for j in np.argsort(-sims)]
            for c in cutoffs:
                sh[c][0]+=int(hit_at(reranked,ev,c)); sh[c][1]+=1
    print("\n=== SEMANTIC RERANK (e5-small) vs HASH (same fusion pool) ===")
    for c in cutoffs:
        print(f"hit@{c}: hash={100*hh[c][0]/hh[c][1]:.2f}%  semantic={100*sh[c][0]/sh[c][1]:.2f}%  ({100*(sh[c][0]-hh[c][0])/hh[c][1]:+.2f})")

if __name__=="__main__": main()
