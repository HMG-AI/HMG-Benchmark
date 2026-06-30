#!/usr/bin/env python3
"""Full query-rewrite fusion retrieval: for each question, GLM generates 3 rewrite
variants; each variant + original does adaptive recall top-50; RRF-fuse into top-150.
Outputs E2E-compatible fused frozen JSON."""
import argparse, asyncio, json, re, sys, time, os
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import hmg_client as lib
from openai import AsyncOpenAI

DOMAIN="software-engineering"; CTX={"tenant_id":"tenant-acme","workspace":"platform","repository":"locomo-benchmark"}
RECALL_CFG=dict(max_results=50,precision_mode=False,disable_adaptive_recall=False,disable_embedding_noise_gate=True,disable_seed_length_penalty=False,disable_lifecycle_ranking=False)
K_RRF=60
CAT={1:"multi-hop",2:"temporal",3:"open-domain",4:"single-hop",5:"adversarial"}
SCORED={1,2,3,4}

def parse_ids(t): return re.findall(r"\[(D\d+:\d+)\]", t)
def ids_set(texts):
    s=set()
    for t in texts:
        for d in parse_ids(t): s.add(d)
    return s
def hit_at(texts,ev,kk): return bool(ids_set(texts[:kk]) & set(ev))
def rrf_fuse(lists,k=K_RRF):
    sc=defaultdict(float); by={}
    for src in lists:
        for rank,text in enumerate(src,1):
            ids=parse_ids(text); key=ids[0] if ids else text[:80]
            sc[key]+=1.0/(k+rank)
            if key not in by: by[key]=text
    return [by[k_] for k_,_ in sorted(sc.items(),key=lambda x:-x[1])]

RW_PROMPT="""You generate search queries for a memory system over past chat conversations.
Given a question, write 5 DIVERSE retrieval queries, each targeting a DIFFERENT facet so they retrieve different memories:
1. People/names involved
2. Events/actions/activities
3. Dates/times/time periods
4. Attributes/descriptions/qualities
5. Places/locations/relationships
Use declarative phrasing matching how memories are stored (e.g. "Caroline visited LGBTQ support group" not "When did Caroline...").
Keep each query under 10 words. Output exactly 5 lines, one per facet, no numbering, no quotes.

Question: {q}"""

class AsyncRateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / max(1, rpm)
        self._lock = asyncio.Lock(); self._next_at = 0.0
    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if now < self._next_at: await asyncio.sleep(self._next_at - now)
            self._next_at = time.monotonic() + self.interval

class Rewriter:
    def __init__(self,key,model="glm-5.2",max_workers=8,rpm=150):
        self.client=AsyncOpenAI(api_key=key,base_url="https://open.bigmodel.cn/api/paas/v4")
        self.model=model; self.sem=asyncio.Semaphore(max_workers); self.lim=AsyncRateLimiter(rpm)
    async def rewrite(self,q):
        async with self.sem:
            await self.lim.wait()
            for attempt in range(6):
                try:
                    r=await self.client.chat.completions.create(model=self.model,
                        messages=[{"role":"user","content":RW_PROMPT.format(q=q)}],
                        max_tokens=160,temperature=0.4,extra_body={"enable_thinking":False})
                    lines=[l.strip().lstrip("0123456789.-) ") for l in (r.choices[0].message.content or "").split("\n") if l.strip()]
                    return [l for l in lines if len(l)>3][:5]
                except Exception as e:
                    msg=str(e)
                    back=min(30, 3*attempt + (10 if "429" in msg or "rate" in msg.lower() else 1))
                    if attempt<5: await asyncio.sleep(back)
            return []

async def batch_rewrite(rw,items):
    async def one(it):
        it["rewrites"]=await rw.rewrite(it["question"]); return it
    return await asyncio.gather(*(one(it) for it in items))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,default=Path("locomo10.json"))
    ap.add_argument("--stores-dir",type=Path,default=Path("benchmark_stores_all"))
    ap.add_argument("--output",type=Path,default=Path("fused_retrieval_top150.json"))
    ap.add_argument("--top-n",type=int,default=150)
    ap.add_argument("--n-rewrites",type=int,default=3)
    ap.add_argument("--limit-conv",type=int,default=None)
    args=ap.parse_args()
    key=os.environ["ZHIPU_API_KEY"]
    ds=lib.load_dataset(args.dataset)
    if args.limit_conv: ds=ds[:args.limit_conv]

    # 1. collect all scored questions
    items=[]
    for ci,s in enumerate(ds):
        for qi,qa in enumerate(s["qa"]):
            if qa["category"] in SCORED:
                items.append({"ci":ci,"qi":qi,"question":qa["question"],"category":qa["category"],
                              "answer":qa.get("answer") or qa.get("adversarial_answer"),
                              "evidence":qa.get("evidence") or []})
    print(f"Rewriting {len(items)} questions...",file=sys.stderr,flush=True)
    rw=Rewriter(key,max_workers=8,rpm=150)
    t0=time.time()
    items=asyncio.run(batch_rewrite(rw,items))
    n_rw=sum(len(it["rewrites"]) for it in items)
    print(f"  {n_rw} rewrites in {time.time()-t0:.0f}s (avg {n_rw/max(1,len(items)):.1f}/q)",file=sys.stderr,flush=True)
    by_cq={(it["ci"],it["qi"]):it for it in items}

    # 2. per-conversation multi-query recall + fusion
    cutoffs=[20,50,100,150]
    agg={c:[0,0] for c in cutoffs}
    bcat=defaultdict(lambda:{c:[0,0] for c in cutoffs})
    results=[]
    t0=time.time()
    for ci,s in enumerate(ds):
        sid=s["sample_id"]; store=args.stores_dir/sid
        ctx=dict(CTX); ctx["branch"]=sid
        qres=[]
        with lib.McpClient(store) as c:
            for qi,qa in enumerate(s["qa"]):
                if qa["category"] not in SCORED: continue
                it=by_cq[(ci,qi)]; ev=it["evidence"]
                queries=[it["question"]]+it["rewrites"][:args.n_rewrites]
                lists=[]
                for q in queries:
                    d=c.call_tool("memory_recall",{"query":q,"context":ctx,"domain_pack_id":DOMAIN,**RECALL_CFG})
                    lists.append([a.get("text","") for a in d.get("atoms",[])])
                fused=rrf_fuse(lists)[:args.top_n]
                for cc in cutoffs:
                    h=hit_at(fused,ev,cc); agg[cc][0]+=int(h); agg[cc][1]+=1
                    bcat[qa["category"]][cc][0]+=int(h); bcat[qa["category"]][cc][1]+=1
                qres.append({"question_index":qi,"question":qa["question"],"category":qa["category"],
                    "category_name":CAT[qa["category"]],"answer":it["answer"],"evidence":ev,
                    "retrieved_count":len(fused),"retrieved_texts":fused,"hit":hit_at(fused,ev,100),"first_hit_rank":None})
        results.append({"sample_id":sid,"conversation_index":ci,"questions":qres})
        print(f"  conv{ci} {sid} done ({time.time()-t0:.0f}s)",file=sys.stderr,flush=True)
    summary={"scored_total":agg[100][1],
        **{f"hit@{c}":agg[c][0] for c in cutoffs},
        **{f"hit_rate@{c}":agg[c][0]/agg[c][1] for c in cutoffs},
        "category_metrics":{str(cat):{f"hit@{c}":bcat[cat][c][0] for c in cutoffs} for cat in sorted(bcat)}}
    payload={"benchmark":"locomo","mode":"query_rewrite_fusion_rrf","top_k":args.top_n,
        "n_rewrites":args.n_rewrites,"results":results,"summary":summary}
    args.output.write_text(json.dumps(payload,ensure_ascii=False))
    print("\n=== FUSION FULL HIT EVAL ===",file=sys.stderr)
    for c in cutoffs: print(f"hit@{c}: {agg[c][0]}/{agg[c][1]} = {100*agg[c][0]/agg[c][1]:.2f}%",file=sys.stderr)
    for cat in sorted(bcat):
        print(f"  {CAT[cat]}: "+" ".join(f"@{c}={100*bcat[cat][c][0]/bcat[cat][c][1]:.1f}%" for c in cutoffs),file=sys.stderr)
    print(str(args.output),file=sys.stderr)

if __name__=="__main__": main()
