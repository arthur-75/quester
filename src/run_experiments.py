#!/usr/bin/env python
"""
Batch-runner for query-reformulation experiments.
Writes/updates results.csv with one row per dataset × system × metric.
"""
#https://castorini.github.io/pyserini/2cr/beir.html
#scp -r satouf@134.157.18.252:/home/satouf/projects/GRPO/checkpoints/Qwen_GRPO_MSMARCO_2.5-1.5M-50000q/checkpoint-750 ./checkpoints/test_on_gpu/
# nfcorpus fiqa scifact nq trec-covid msmarco webis-touche2020 climate-fever
import argparse
import os
import random
from pathlib import Path

import pandas as pd
torch_available = False
import torch
if torch.cuda.is_available():
    torch_available = True

from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from utils import (
    get_data,
    creat_index,
    generate_reformulations,
    eval_reformulations,
)
import yaml
import sys

# ---------- YAML ARGUMENT PATCHING ----------
# If first arg ends in .yaml, load and override sys.argv
if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml"):
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f)
    sys.argv = [sys.argv[0]]  # Reset to script name
    for k, v in cfg.items():
        if isinstance(v, list):
            sys.argv += [f"--{k}"] + list(map(str, v))
        elif isinstance(v, bool):
            if v:
                sys.argv.append(f"--{k}")
        else:
            sys.argv += [f"--{k}", str(v)]


# ---------- helpers ----------------------------------------------------------
def sample_dict(d, k=5000, seed=4):
    """Return a new dict with k random entries from d (deterministic)."""
    if len(d) <= k:
        return d
    random.seed(seed)
    keys = random.sample(list(d.keys()), k)
    return {key: d[key] for key in keys}


def ensure_index(corpus, index_path):
    """Create Lucene index if it doesn't exist yet."""
    if Path(index_path, "segments.gen").exists():
        return
    print(f"[index] building → {index_path}")
    creat_index(index_path, corpus)


def run_model(model, tokenizer, queries,system_prompt,dataset,keywords_corpus=None, batch_size=64, temp=.8,output_keywords=True,max_new_tokens=64):
    """Generate reformulations for all queries."""
        
    # For  original base model:
    generations = generate_reformulations(
        queries=queries,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        system_prompt=system_prompt,
        dataset=dataset,
        keywords_corpus=keywords_corpus,
        temperature=temp,batch_size=batch_size,max_new_tokens=max_new_tokens,
        output_keywords=output_keywords)
    return  generations


# ---------- main -------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="List of dataset names (get_data compatible)")
    ap.add_argument("--data_path", default="data/text_data/")
    ap.add_argument("--index_root", default="data/index_data/")
    ap.add_argument("--grpo_ckpt", required=True,
                    help="Path to GRPO fine-tuned checkpoint")
    ap.add_argument("--outfile", "--out", default="data/results/results")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--temp", type=float, default=.8)
    ap.add_argument("--base_out", action="store_true", help="Enable base output mode")
    ap.add_argument("--system_prompt", default="You are an assistant Rewriter. Output ONLY the new query with extra relevant vocabulary.")
    ap.add_argument("--base_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--keywords_corpus", action="store_true", help="Enable keywords corpus")
    ap.add_argument("--save_query", action="store_true", help="Save generated queries")
    ap.add_argument("--output_keywords", action="store_true", help="output keywords or a query")
    ap.add_argument("--add_bm25_query", action="store_true", help="to add query to generated query")
    ap.add_argument("--beta", type=int, default=3)
    ap.add_argument("--n_rep", type=int, default=1)
    ap.add_argument("--surfix", type=str,default="")
    ap.add_argument("--count_time", action="store_true",  help="count time computation ")
    ap.add_argument("--need_bm25", action="store_true", help="Enable bm25 output")
    ap.add_argument("--batch_threads", type=int, default=35)
    args = ap.parse_args()

    # ---------- load base & GRPO models once -------------------------------
    df_prev = None
    main_path=Path('data/results/main.csv')
    if main_path.exists():
        print("there is a main path")
        df_prev = pd.read_csv(main_path)

    if args.base_out or main_path.exists() :
        print("[model] loading base model …")
        base_tok = AutoTokenizer.from_pretrained(args.base_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_name, torch_dtype="auto", device_map="auto"
        )

    print("[model] loading GRPO checkpoint …")
    grpo_tok = AutoTokenizer.from_pretrained(args.grpo_ckpt)
    grpo_model = AutoModelForCausalLM.from_pretrained(
        args.grpo_ckpt, torch_dtype="auto", device_map="auto"
    )
   
       

    rows = []
    batch_threads=args.batch_threads
    for ds in args.datasets:
        batch_size=args.batch
        print(f"\n=== Dataset: {ds} ===")
        no_bier=False
        split="test"
        if ds== "msmarco":split="dev"
        if ds=="arguana":
            batch_size=32
            batch_threads=1
        if Path(args.data_path+"/"+ds).exists():no_bier=True
        corpus, queries, qrels = get_data(ds, data_path=args.data_path, split=split,no_bier=no_bier)
        if len(queries)>2000:batch_threads=1
        # sub-sample if too large
        #if len(queries) > 20000:
        #    queries = sample_dict(queries, k=5000)
        queries_ids = list(queries.keys())

        # build index
        index_path = Path(args.index_root, f"{ds}_docIndex")
        if ds == "trec-dl-2019"or ds== "trec-dl-2020" : index_path=   Path(args.index_root, f"msmarco_docIndex")
        ensure_index(corpus, index_path)
        searcher = LuceneSearcher(str(index_path))
        searcher.set_bm25(k1=0.9, b=0.4)

        
        indices = random.sample(range(len(queries)), 5)
        # --- BM25 baseline ------------------------------------------------
        
        need_bm25 = True
        if df_prev is not None:
            if ((df_prev["dataset"] == ds) & (df_prev["system"] == "BM25")).any():
                print("→ BM25 already exists in CSV, skipping recompute")
                need_bm25 = False
        datas=[]
        bm_list=list(queries.values())
        if need_bm25 or args.need_bm25:
            print("→ BM25 …")
            bm_metrics = eval_reformulations(
                bm_list, queries_ids, qrels, searcher=searcher,batch_threads=batch_threads,
                name_model="BM25",
                dataset=ds,
                count_time=args.count_time,
            )
            
            for metric_dict in bm_metrics:
                for name, val in metric_dict.items():
                    data={
                        "dataset": ds,
                        "system": "BM25",
                        "metric(%)": name,
                        "value": val*100
                    }
                    rows.append(data)
                    datas.append(data)
        print('')
        print("## 5 queries randomly selected from BM25", )
        for i in indices:
            print(f"- {list(queries.values())[i]} ")
        # --- Base LLM -----------------------------------------------------

        need_qwen = True
        if df_prev is not None:
            if ((df_prev["dataset"] == ds) & (df_prev["system"] == "Qwen-base")).any():
                if not  args.base_out :
                    print("→ Qwen-base already exists in CSV, skipping recompute")
                    need_qwen = False

        if args.base_out or need_qwen:
                print("→ Qwen-base …")
                base_out = run_model(base_model, base_tok, queries,system_prompt=args.system_prompt,
                                    batch_size=batch_size,temp=args.temp,dataset=ds,keywords_corpus=args.keywords_corpus, 
                                    output_keywords=args.output_keywords,max_new_tokens=128
                                    )s
                
                base_metrics = eval_reformulations(
                    base_out, queries_ids, qrels, searcher=searcher,batch_threads=batch_threads
                )
                for metric_dict in base_metrics:
                    for name, val in metric_dict.items():
                        data={
                            "dataset": ds,
                            "system": "Qwen-base-sft",
                            "metric(%)": name,
                            "value": val*100
                        }
                        rows.append(data)
                        datas.append(data)
                print("## 5 queries randomly selected from Base-model", )
                for i in indices:
                    print(f"- {base_out[i]} ")

        # --- GRPO LLM -----------------------------------------------------
        print("→ GRPO …")
        all_grpo_out=[]
        temp=args.temp
        for i in range(args.n_rep):
        
            grpo_out = run_model(grpo_model, grpo_tok, queries,system_prompt=args.system_prompt,
                                batch_size=batch_size
                                ,temp=temp,dataset=ds,keywords_corpus=args.keywords_corpus, 
                                output_keywords=args.output_keywords,
                                )
            temp=args.temp
            if len(all_grpo_out)==0: all_grpo_out= grpo_out.copy()
            else : all_grpo_out= [all_ +" "+gr_ for all_ ,gr_ in zip(all_grpo_out,grpo_out) ]

        grpo_out = all_grpo_out
        beta_muja=0
        if args.add_bm25_query and args.n_rep >1:

            beta_muja=args.beta
            grpo_out=[(b+" ")*int(max(1, (len(g)/(len(b)* beta_muja))))+" "+ g for g,b in zip(grpo_out,bm_list)]
        elif  args.add_bm25_query:  
            beta_muja=1
            grpo_out=[b+" "+ g for g,b in zip(grpo_out,bm_list)]
        
        #*int(max(1, (len(g)/(len(b)*4))))
        if args.save_query:
            name_q=f"data/queries_by_grpo/{ds}-{args.grpo_ckpt.split("/")[-1]}-{args.grpo_ckpt.split("/")[-2]}.txt"
            with open(name_q, "w", encoding="utf-8") as f:
                for line in grpo_out:
                    f.write(line + "\n")
        
        grpo_metrics = eval_reformulations(
            grpo_out, queries_ids, qrels, searcher=searcher,batch_threads=batch_threads,
            name_model=f"GRPO-{(args.grpo_ckpt.split("/")[-2])}",
                dataset=ds,
                count_time=args.count_time,
        )
        for metric_dict in grpo_metrics:
            for name, val in metric_dict.items():
                data={
                    "dataset": ds,
                    "system": "GRPO",
                    "metric(%)": name,
                    "value": val*100
                }
                rows.append(data)
                datas.append(data)
        print("## 5 queries randomly selected from GRPO", )
        for i in indices:
            print(f"- {grpo_out[i]} ")

        df = pd.DataFrame(datas)
        pivot = df.pivot_table(
        index=["dataset", "system"],
        columns="metric(%)",
        values="value"
        ).round(1).reset_index()
        print(f"\n=== Aggregated results - {ds} ===")
        if df_prev is not None:
            df_all = pd.concat([df_prev, pivot], ignore_index=True)
            #df_all = df_all.drop_duplicates(subset=["dataset", "system", "metric(%)"])
            pivot = df_all[df_all["dataset"] == ds].reset_index(drop=True)
            print(pivot)
        else:
            print(pivot.reset_index(drop=True))

    # ---------- save / append to CSV ----------------------------------------
    df = pd.DataFrame(rows)
    Path(args.outfile+f"/{args.grpo_ckpt.split("/")[-2]}").mkdir(parents=True, exist_ok=True)
    outpath = Path(args.outfile+f"{"/".join(args.grpo_ckpt.split("/")[-2:])}_temp-{args.temp}_nq{args.n_rep}-mu{beta_muja}{args.surfix}.csv")
   
    #df = df.drop_duplicates(subset=["dataset", "system","metric(%)"]).reset_index(drop=True)
    # ---------- pretty print -------------------------------------------------
    #df_final = pd.read_csv(outpath)
    pivot = df.pivot_table(
        index=["dataset", "system"],
        columns="metric(%)",
        values="value"
    ).round(1).reset_index()
    if df_prev is not None:
        pivot = pd.concat([df_prev, pivot], ignore_index=True)
    print(outpath)
    pivot.reset_index(drop=True).sort_values(by=["dataset","system"]).to_csv(outpath, index=False)



if __name__ == "__main__":
    main()
