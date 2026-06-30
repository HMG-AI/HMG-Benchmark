#!/usr/bin/env python3
"""RRF-fuse a fusion-retrieval JSON with the original real-embedding JSON to see if
adding real-embedding recall boosts hit@k further. Zero retrieval cost (JSON only)."""
import argparse, json, re
from collections import defaultdict
from pathlib import Path
SCORED={1,2,3,4}; CAT={1:"multi-hop",2:"temporal",3:"open-domain",4:"single-hop",5:"adversarial"}
def parse_ids(t): return re.findall(r"\[(D\d+:\d+)\]", t)
def ids_set(texts):
    s=set()
    for t in texts:
        for d in parse_ids(t): s.add(d)
    return s
def hit_at(texts,ev,kk): return bool(ids_set(texts[:kk]) & set(ev))
def rrf_fuse(lists,k=60):
    sc=defaultdict(float); by={}
    for src in lists:
        for rank,text in enumerate(src,1):
            ids=parse_ids(text); key=ids[0] if ids else text[:80]
            sc[key]+=1.0/(k+rank)
            if key not in by: by[key]=text
    return [by[k_] for k_,_ in sorted(sc.items(),key=lambda x:-x[1])]
def idx(d):
    m={}
    for c in d["results"]:
        for q in c["questions"]:
            m[(int(c["conversation_index"]),int(q["question_index"]))]=q
    return m
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fusion",type=Path,required=True)
    ap.add_argument("--orig",type=Path,default=Path("frozen_hmg_retrieval.json"))
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--top-n",type=int,default=150)
    args=ap.parse_args()
    fu=idx(json.loads(args.fusion.read_text())); og=idx(json.loads(args.orig.read_text()))
    cutoffs=[20,50,100,150,200]
    single={c:[0,0] for c in cutoffs}; dual={c:[0,0] for c in cutoffs}
    bcat=defaultdict(lambda:{c:[0,0] for c in cutoffs})
    fused_out=[]
    for key in sorted(fu):
        q=fu[key]; cat=int(q["category"])
        if cat not in SCORED: continue
        ev=q.get("evidence") or []
        fu_texts=q.get("retrieved_texts") or []
        og_q=og.get(key); og_texts=(og_q.get("retrieved_texts") or [])[:20] if og_q else []
        dual_texts=rrf_fuse([fu_texts, og_texts])[:args.top_n]
        for c in cutoffs:
            single[c][0]+=int(hit_at(fu_texts,ev,c)); single[c][1]+=1
            dual[c][0]+=int(hit_at(dual_texts,ev,c)); dual[c][1]+=1
            bcat[cat][c][0]+=int(hit_at(dual_texts,ev,c)); bcat[cat][c][1]+=1
        if args.output: fused_out.append((key,q,dual_texts))
    print("=== DUAL-SOURCE (fusion + orig-real-embedding) HIT ===")
    for c in cutoffs:
        s,d=single[c],dual[c]
        if s[1]: print(f"hit@{c}: fusion_only={100*s[0]/s[1]:.2f}%  dual={100*d[0]/d[1]:.2f}%  (+{100*(d[0]-s[0])/s[1]:.2f})")
    print("dual by category:")
    for cat in sorted(bcat):
        print(f"  {CAT[cat]}: "+" ".join(f"@{c}={100*bcat[cat][c][0]/bcat[cat][c][1]:.1f}%" for c in cutoffs))
    if args.output:
        by_conv=defaultdict(list)
        for key,q,texts in fused_out: by_conv[key[0]].append((key[1],q,texts))
        results=[]
        for ci in sorted(by_conv):
            qs=[]
            for qi,q,texts in sorted(by_conv[ci]):
                topn=texts[:args.top_n]
                qs.append({"question_index":qi,"question":q["question"],"category":int(q["category"]),
                    "category_name":CAT[int(q["category"])],"answer":q.get("answer"),"evidence":q.get("evidence") or [],
                    "retrieved_count":len(topn),"retrieved_texts":topn,"hit":hit_at(topn,q.get("evidence") or [],100),"first_hit_rank":None})
            results.append({"sample_id":q.get("sample_id",f"conv-{ci}"),"conversation_index":ci,"questions":qs})
        payload={"benchmark":"locomo","mode":"dual_fusion_rrf","sources":[str(args.fusion),str(args.orig)],
                 "top_k":args.top_n,"results":results}
        args.output.write_text(json.dumps(payload,ensure_ascii=False))
        print(f"wrote {args.output}")
if __name__=="__main__": main()
