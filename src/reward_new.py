# reward_class.py
from typing import Dict, List, Tuple
import os, re
import logging
import numpy as np
from datetime import datetime

import itertools

def len_reward(query_text):
    long_row=0.25
    l_text=len(query_text)
    min_num_t=50
    if l_text <= min_num_t:
        long_row=l_text/40
    return long_row /7


class BaseReward:
    """
    Minimal class wrapper around your base_reward() function so you can tune
    parameters in __init__ without touching the core logic.
    """
    def __init__(
        self,
        ce_tsv_path: str = "data/CE_data/dictCE.tsv",
        top_k: int = 100,
        #ce_k: int = 100,
        tau: float = 0.5,
        threads: int = 30,
        log_dir: str = "logs",
        debug_name: str = "sft_data_generation",
        log_discounts_cap: int = 10000,
        add_init=False,
        searcher=None,
        qrels=None,
    ):
        self.ce_tsv_path = ce_tsv_path
        self.top_k = top_k
        #self.ce_k = ce_k
        self.tau = tau
        self.threads = threads
        self.add_init=add_init
        self.searcher=searcher

        # logger (file-only)
        log_path = os.path.join(log_dir, debug_name)
        os.makedirs(log_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("reward")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(log_path, f"{debug_name}_{timestamp}.log"), mode="w")
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        # load CE dict once
        self.dict_ce: Dict[str, Dict[str, int]] = {}
        with open(self.ce_tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                qid, _, docid, rel = line.strip().split("\t")
                rel = int(rel)
                bucket = self.dict_ce.setdefault(qid, {})
                bucket[docid] = rel
        print("reward_new")
        # precompute discounts once
        self.log_discounts = np.log2(2 + np.arange(log_discounts_cap))

    # ---------- helper: exact E[ 1/log2(2+rank) ] ----------
    def _soft_discount_full(self, pi_row: np.ndarray,log_discounts) -> float:
        k = len(pi_row)
        p = np.zeros(k, dtype=float)
        p[0] = 1.0
        for pij in pi_row:
            if pij == 0.0:
                continue
            p[1:] = p[1:] * (1 - pij) + p[:-1] * pij
            p[0] *= (1 - pij)
        #log_discounts=self.log_discoufselfnts[:k]
        #return float((p * (1 /log_discounts)).sum())
        return float((p /log_discounts).sum())

    # ---------- SoftRank NDCG (your version) ----------
        
        # ---------- SoftRank NDCG (your version) ----------
    def _soft_ndcg(
            self,
            scores,
            ce_bm25,
            ce_qrels,
            k_soft: int=100,
            tau: float=0.5
        ) -> float:
            
            #soft_list = [(d, s) for d, s in scores.items() if d not in ce_bm25][:k_soft]
            top_list = dict(itertools.islice(scores.items(), k_soft))
            ce_bm25 = dict(sorted(ce_bm25.items(), key=lambda item: item[1],reverse=True))
            missed= []
            for k,v in ce_bm25.items():
                #if k in scores :
                    if k not in top_list :
                        top_list[k]=v
                        missed.append(str(k))
                        
            
            # Filter: only consider missed docs that are relevant and have positive BM25 score
            
            rank_map = {d:i for i,d in enumerate(scores.keys())} 

            missed_ranked= [rank_map[i]   for i in missed ]


        
            s_soft = np.array(list(top_list.values())) #np.array([v for _, v in top_list], dtype=float)
            diff = (s_soft[:, None] - s_soft[None, :]) / tau
            pi = 1.0 / (1.0 + np.exp(-diff))
            np.fill_diagonal(pi, 0.0)
            

            log_discounts = np.concatenate(( self.log_discounts[:len(top_list)-len(missed_ranked)], [ self.log_discounts[rank_r ]  for rank_r in missed_ranked] ) )

            discount_soft = np.fromiter((self._soft_discount_full(r,log_discounts) for r in pi.T), float)
            g_soft = np.array([ce_qrels.get(d, 0.0) for d in top_list.keys()], dtype=float)
            sdcg = float((g_soft * discount_soft).sum())



            idcg = float((np.sort(g_soft)[::-1] / log_discounts ).sum())
            
            return sdcg / (idcg + 1e-9)


    # ---------- main callable ----------
    def __call__(
        self,
        completions,
        queries_id,
        searcher,
        reader,
        queries,
        qrels=None,
        DEBUG_QID=None,
        CrossEn=None,
        reRanker=None,
        corpus=None,
    ) -> List[float]:

        DEBUG_QID = set(DEBUG_QID or [])
        gen_texts = [c[0]["content"] for c in completions]
        if self.add_init:
            gen_texts =[gen_text +" "+queries[query_id] for gen_text, query_id in zip(gen_texts, queries_id) ]

        batch_qids = [str(i) for i in range(len(queries_id))]
        hits = self.searcher.batch_search(
            gen_texts,
            batch_qids,
            k=self.top_k,
            threads=self.threads
        )

        rewards: List[float] = []
        debug_buffer = []
        
        for i, qid in enumerate(batch_qids):
            qid_actual = queries_id[i]
            query_text = gen_texts[i]

            # CE docids limited to ce_k
            ce_pool = self.dict_ce.get(qid_actual, {})
            ce_docids = list(ce_pool.keys())[:self.top_k]

            # reference scores from retrieval
            scores_ref = {d.docid: d.score for d in hits[qid] if d.docid != str(qid_actual)}


            #ce_bm25 = {d: reader.compute_query_document_score(d, query_text) for d in ce_docids}
            #bm_max_qr = max(ce_bm25.values()) 

            ce_bm25 = {d:scores_ref[d] for d in ce_docids if d in scores_ref }

            if  not ce_bm25 :
                rewards.append(0.0)
                if qid_actual in DEBUG_QID:
                    debug_buffer.append((i, qid_actual, f"[bm=0] Generated: {query_text}"))
                continue

            ce_qrel = {d: ce_pool[d] / 1000.0 for d in ce_docids}

            

            # main reward (SoftRank NDCG)
            soft_ndcg = self._soft_ndcg(scores_ref, ce_bm25, ce_qrel, tau=self.tau)
            
            #len_rew=len_reward(query_text)
            reward = soft_ndcg#+len_rew
            rewards.append(reward+len_reward(query_text))

            if qid_actual in DEBUG_QID:
                #relevant_docids = list(qrels[qid_actual].keys())
                #orgi = np.max([reader.compute_query_document_score(docid, queries[qid_actual]) for docid in relevant_docids]) if relevant_docids else 0.0
                bm_list = [f"{v:.1f}" for v in list(scores_ref.values())[:3]]
                #top_doc_text = ""
                #if scores_ref and corpus:
                first_id = next(iter(scores_ref.keys()))
                #top_doc_text = (corpus[first_id]["text"][:200]) #if first_id in corpus else ""
                log_msg = (
                    f"[{i + 1:03d}] Query ID: {qid_actual}"
                    f" - BM q_gen-top3: {bm_list}\n"
                    #f" - BM q0-ref: {orgi:.4f}"
                    #f" - BM q_gen-ref: {bm_max_qr:.4f}"
                    f" - SoftRank@{self.top_k}: {soft_ndcg:.4f}"
                    f" - Reward: {reward:.4f}\n"
                    f" - Reference Query: {queries[qid_actual]}\n"
                    f" - Generated Query: {query_text}\n"
                    #f" - Top Retrieved Doc Text:  \"{top_doc_text}...\"\n"
                    f" - Top Doc id:  {str(first_id)}"
                )
                debug_buffer.append((i, qid_actual, log_msg))

        # write debug logs at the end (kept identical in spirit)
        try:
            for i, qid, msg in debug_buffer:
                self.logger.debug("%s Final Reward: %.4f\n%s", msg, rewards[i], "-" * 80)
        except Exception as e:
            self.logger.debug("Debug logging failed: %s", str(e))

        return rewards
#
