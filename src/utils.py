#1570
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import subprocess
import json
from tqdm import tqdm
import random
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from datasets import Dataset
import torch
from langdetect import detect, detect_langs
from beir.retrieval.evaluation import EvaluateRetrieval
import pickle
from typing import Dict, List
import numpy as np
import pytrec_eval
from scipy.stats import wilcoxon
import os, pickle, gzip
from typing import Dict
import time, json

def make_conversation(example,SYSTEM_PROMPT,output_keywords=True):
    #keywords_corpus=f" [HINT] Words extracted from the corpus; inspiration only, avoid direct use: {keywords_corpus[example["queries_id"]]}." if keywords_corpus else ""
    if output_keywords:
        return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":"[QUERY]: "+ example["prompt"]+" [KEYWORDS]:"},
        ], 
        }
    return {
        
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content":"Generate keywords separated by commas that might appear in relevant documents to the following [QUERY]: "+ example["prompt"]+" [KEYWORDS]:"}, 
        ], 
    }

def get_data(data_set, data_path="data",split="test",no_bier=True):
    if not no_bier:
        print("no bier")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
            data_set
        )
        data_path = util.download_and_unzip(url, data_path)
    else:data_path=data_path+data_set
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    # queries_ids = list(queries.keys())
    # queries= list(queries.values())
    # documents = [[f"{doc['title']} ,{doc['text']}"] for doc in corpus.values()]
    # document_ids= list(corpus.keys())

    return corpus, queries, qrels

def creat_index(index_path,corpus):
    
    if not os.path.isdir(index_path):
      os.makedirs(index_path)
    else :return None
    pyserini_jsonl = "pyserini.jsonl"
    # Build the command
    command = [
    "python", "-m", "pyserini.index.lucene",
    "--collection", "JsonCollection",
    "--input", index_path,
    "--index", index_path,
    "--generator", "DefaultLuceneDocumentGenerator",
    "--threads", "1",
    "--storePositions", "--storeDocvectors", "--storeRaw"   
    ]   

    with open(os.path.join(index_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in  tqdm(corpus):
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id , "contents": title+" "+text}
            json.dump(data, fOut)
            fOut.write('\n')
    subprocess.run(command)
    return  None

from types import MethodType

def apply_chat_template_no_think(self, *args, **kwargs):
    # Force the keyword unless the caller already set it
    kwargs.setdefault("enable_thinking", False)
    kwargs.setdefault("add_generation_prompt", True)
    return self.__class__.apply_chat_template(self, *args, **kwargs)


def sample_dict(d, num_train=1000, seed=42):
    random.seed(seed)
    keys = list(d.keys())
    num_train= min(num_train,len(keys))
    sampled_keys = random.sample(keys, num_train)
    return {key: d[key] for key in sampled_keys}


def get_train_dataset(dataset,data_path, index_path,num_train,  SYSTEM_PROMPT,seed ,keywords_corpus=None,split='train',output_keywords=True):
    corpus, queries, qrels = get_data(dataset, data_path, split=split)
    #queries = sample_dict(queries, num_train=num_train, seed=seed)
    
    # Build index only from filtered corpus
    index_path=index_path+f"{dataset}_docIndex"
    creat_index(index_path, corpus)
    ce_qrel=None
    corpus= { k:{"text":v["text"][:500]} for k,v in corpus.items()}
    #with open("data/CE_data/CE_q_1.pkl", "rb") as f:
    #    ce_qrel= pickle.load(f)
    
    """#select quries 
    long_queries={}
    for key,value in queries.items():
        if len(value)>45:
            long_queries[key]=value

    long_cor_queires={}
    for key,value in long_queries.items():
        qu=qrels[key]
        for i in qu.keys():
            if len(corpus[i]["text"])>340:
                #print(corpus[i])
                long_cor_queires[key]=value
                break
    queries=long_cor_queires"""
    #queries={ k:v for k,v in queries.items() if k in ce_qrel}
    # Set up retriever
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1=0.9, b=0.4)
    reader= LuceneIndexReader(index_path)
    # Format dataset for training
    
    if keywords_corpus:
        path = Path(f"data/bm10_keywords/{dataset}_key10.pkl")
        with path.open("rb") as f:
            keywords_corpus = pickle.load(f)   # dict: {query_id: "k1, k2, ..."}
        
    dataset = Dataset.from_list(
    [{"prompt":value,"queries_id":key}  for key, value in queries.items()
    ]
    )
    train_dataset = dataset.map(lambda x: make_conversation(x, SYSTEM_PROMPT,keywords_corpus,output_keywords))
    print(train_dataset[0])


        


    # Keep only docs in qrels
    """docids_in_qrels = set()
    for doc_dict in qrels.values():
        docids_in_qrels.update(doc_dict.keys())
    corpus = {docid: doc for docid, doc in corpus.items() if docid in docids_in_qrels}
    """
    # Debug subset of queries
    print("number of queries ",len(queries))
    ex_random=min(len(queries),100000)
    DEBUG_QID=random.sample(list(queries.keys()),ex_random)
    
    return train_dataset.select(range(92800)), queries,qrels,searcher,reader,DEBUG_QID,None,ce_qrel



# ---------- Utilities -------------------------------------------------
def fix_torch_seed(seed=30):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import logging, sys, os
from pathlib import Path

def setup_logging(output_dir, level=logging.DEBUG):
    """Create <output_dir>/train.log and mirror everything to stdout."""
    log_path = Path(output_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "[%(asctime)s] [%(levelname)-5s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Silence tens of INFO lines from transformers if you wish:
    logging.getLogger("transformers").setLevel(logging.WARNING)



def eval_reformulations(
    completions: list[str],
    query_ids: list[str],
    qrels,
    searcher=None,
    k: int = 100,
    only_search=False,
    batch_threads=None,
    name_model="",
    dataset="",
    count_time=False,
):
    """
    Evaluate reformulated queries using NDCG@{1,10,100}.
    Optionally compares against the original queries.

    Args:
        completions: List of reformulated query strings.
        query_ids: List of corresponding query IDs.
        queries: Original queries dictionary (optional).
        searcher: Pyserini LuceneSearcher instance.
        k: Number of retrieved documents.

    Returns:
        Tuple of:
            - ndcg, recall, precision for reformulated
            - ndcg, recall, precision for original (if `queries` is provided)
    """
    
    scores_ref = {}
    if batch_threads>1:
        hits_all = searcher.batch_search(
            completions, query_ids, k=k, threads=batch_threads,# adjust threads for speed
        )
        
        for qid in query_ids:
            hits = hits_all[qid]
            scores_ref[qid] = {
                d.docid: d.score for d in hits if d.docid != str(qid)
            }
            

    else:
        n = len(query_ids)
        t0 = time.perf_counter()
        for qid, query in zip(query_ids, completions):
            hits = searcher.search(query, k=k)
            scores_ref[qid] = {
                d.docid: d.score for d in hits if d.docid != str(qid)
            }
        t1 = time.perf_counter()
        if count_time:
            total_sec = t1 - t0
            ms_per_query = (total_sec / n * 1000.0) if n else None
            qps = (n / total_sec) if total_sec > 0 else None
            # save timing summary
            out = {
                "dataset": dataset,
                "model": name_model,
                "queries": n,
                "seconds": round(total_sec, 6),
                "ms_per_query": None if ms_per_query is None else round(ms_per_query, 3),
                "QPS": None if qps is None else round(qps, 3),
            }
            fname = f"data/time/{dataset}__{name_model}__timing.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    
    with open(f"data/save_res/{dataset}_{name_model}.pkl", "wb") as f:
            pickle.dump(scores_ref, f, protocol=pickle.HIGHEST_PROTOCOL)

   



    if only_search:
        return scores_ref


    return _eval(scores_ref, qrels)

    

def _eval(scores_ref,qrels):
    # Evaluate reformulated queries
    ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, scores_ref, [1, 10, 100,1000])

    # Evaluate original queries if provided
    mrr = EvaluateRetrieval.evaluate_custom(qrels, scores_ref, metric="mrr", k_values=[10,100])

    return ndcg, recall, mrr



def generate_reformulations(
    queries: Dict[str, str],
    model,
    tokenizer,
    device: str,
    system_prompt: str,
    dataset:str,
    keywords_corpus:bool=None,
    max_new_tokens: int = 128,
    do_sample: bool = True,
    temperature: float = 0.9,
    top_k: int = 500,
    top_p = 0.95,
    batch_size: int = 32,          # 🔁 how many prompts per GPU batch
    output_keywords=True,
    repetition_penalty=1.2
    #n_rep: int = 1,    # 🔢 NEW: how many reformulations per query
) -> List[str]:
    """
    Batched generation that returns one reformulation per query (same order).
    Prompts and <think> blocks are NOT included in the final returned strings.
    """
    # ---------- one-time tokenizer safety ----------
    if temperature==0 :
        do_sample=False
        top_k=None
        top_p=None
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id



    query_texts   = list(queries.values())
    query_ids   = list(queries.keys())
    reformulations: List[str] = []
    

    # ---------- iterate in GPU-friendly chunks ----------
    latencies = []
    for i in tqdm(range(0, len(query_texts), batch_size), desc="Generating in batches"):
        batch_queries = query_texts[i:i + batch_size]
        batch_queries_ids = query_ids[i:i + batch_size]

        # 1) Build prompts (string form, no thinking)
        full_prompts: List[str] = []
        for id_q,q in zip(batch_queries_ids,batch_queries):
            example   = {"prompt": q,"queries_id":id_q}
            conv      = make_conversation(example, system_prompt,output_keywords=output_keywords
                                          )
            rendered  = tokenizer.apply_chat_template(
                conv["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False          # 👈 ensure no <think> tag
                
            )
            full_prompts.append(rendered)
        # 2) Tokenize as one padded batch and push to device
        inputs = tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            #add_special_tokens=false,
        ).to(device)

        # 3) Generate as a batch
        with torch.inference_mode():
            start = time.time()
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                return_dict_in_generate=True,
                repetition_penalty=repetition_penalty,
                output_scores=False,
                top_p = top_p ,
            )
            latencies.append(time.time() - start)

        # 4) Slice off the prompt part & decode
        #new_tokens = gen_out.sequences[:, inputs["input_ids"].shape[1]:]
        #decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
       
        #for text in (decoded):
                #for  text in (decoded):
        seqs   = gen_out.sequences           # (B, prompt+completion)
        prompt_lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
        for idx, seq in enumerate(seqs):
            new_tokens = seq[prompt_lens[idx]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
           

            # drop any leftover </think> or <think> blocks
            if "\nassistant\n" in text:
                text = text.split("\nassistant\n", 1)[-1].lstrip()
            elif "\nassistant" in text:                # safety if the BOS template inserts it
                text = text.split("\nassistant", 1)[-1].lstrip()
            elif "assistant\n" in text:                # safety if the BOS template inserts it
                text = text.split("assistant\n", 1)[-1].lstrip()
            if "</think>\n\n" in text:
                text = text.split("</think>\n\n", 1)[-1].lstrip()
            if "</think>\n" in text:                # safety if the BOS template inserts it
                text = text.split("</think>\n", 1)[-1].lstrip()
            if "</think>" in text:                # safety if the BOS template inserts it
                text = text.split("</think>\n", 1)[-1].lstrip()
            if "<think>" in text:                # safety if the BOS template inserts it
                text = text.split("<think>\n", 1)[-1].lstrip()
            text=text.replace("Keywords:"," ")#.replace("think>"," ")#.replace("Query:","    ")
            reformulations.append(text )
    print("Avg LLM generation time:", sum(latencies)/len(latencies))
    return reformulations





# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

class reRanker_class:

    def __init__(self,model_name="Qwen/Qwen3-Reranker-0.6B",batches=20):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.batches=batches

    def format_instruction(self,instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self,pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self,inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

        
    def rank(self, queries, documents):
        task = 'Given a web search query, retrieve relevant passages that answer the query'
        all_scores = []
        # Process in batches of size n
        for i in range(0, len(queries), self.batches):
            batch_q = queries[i:i+ self.batches]
            batch_d = documents[i:i+ self.batches]
            pairs = [self.format_instruction(task, q, d) for q, d in zip(batch_q, batch_d)]

            # Tokenize and score
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
            
            all_scores.extend(scores)

        return all_scores