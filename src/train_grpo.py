#!/usr/bin/env python
"""Train a query-reformulation model with GRPO + LoRA."""
import argparse, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,PeftModel,PeftConfig
from trl import GRPOTrainer, GRPOConfig
from src.utils import get_train_dataset, apply_chat_template_no_think,fix_torch_seed,setup_logging#,reRanker_class
#from src.rewardSoft2 import base_reward
#from src.reward_new100 import BaseReward
import torch
from types import MethodType
#/data/zong/query_rewrite/xp/sft_data_generation/jobs/learning.sft.sftqueryrewritelearner/1dedceebfb0d607b63c5ede1710641634200edb8210553f53cb7740e0246b679/best/RR@10
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
import gc
gc.collect()

def load_reward(item: str):
    if item == "reward_hard_ce":
        from src.reward_hard_ce import BaseReward
        return BaseReward
    elif item == "reward_hard_new":
        from src.reward_hard_new import BaseReward
        return BaseReward
    elif item == "reward_new":
        from src.reward_new import BaseReward
        return BaseReward
    elif item == "reward_hard":
        from src.reward_hard import BaseReward
        return BaseReward
    elif item == "reward_new100":
        from src.reward_new100 import BaseReward
        return BaseReward
    elif item == "reward_soft_tr":
        from src.reward_soft_tr import BaseReward
        return BaseReward
    else:
        raise ValueError(f"Unknown reward '{item}'")

# ---------- Argument parsing -----------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                   help='Path to YAML config')
    p.add_argument('--resume', action='store_true',
                   help='Resume from last checkpoint')
    # override examples
    p.add_argument('--learning_rate', type=float)
    p.add_argument('--epochs', type=int)
    return p.parse_args()

# ---------- Main ------------------------------------------------------
def main():
    args = parse_args()
    import yaml; cfg = yaml.safe_load(open(args.config))
    cfg.update({k: v for k, v in vars(args).items() if v is not None})

    # ---- Environment --------------------------------------------------
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    fix_torch_seed()
    BaseReward = load_reward(cfg['reward']['load_reward'])

    # ---- Data ---------------------------------------------------------
    train_dataset, queries,qrels,searcher,reader,DEBUG_QID,corpus,ce_qrel=get_train_dataset(cfg['dataset'],cfg['data_path'],
                                cfg['index_path'],cfg["num_train"],cfg['SYSTEM_PROMPT'],cfg["seed"],cfg['keywords_corpus'],
                                cfg['split'],output_keywords=cfg['output_keywords'])
    reader=None
    print("We have the data")
    # ---- Model --------------------------------------------------------
    # inside main() – before you build the model
    setup_logging(cfg['grpo']['output_dir']) 
    processing_class = AutoTokenizer.from_pretrained(cfg['model_name'], padding_side="left",)

    # monkey-patch just this copy
    processing_class.apply_chat_template = MethodType(apply_chat_template_no_think,
                                                    processing_class)
    device_map="auto"
    if torch.cuda.device_count() >1 :                                               
        device_map="balanced"

    if cfg['model_name'][0].lower()=="g":
        base = AutoModelForCausalLM.from_pretrained(cfg['model_name'], attn_implementation="eager",
                                               torch_dtype='auto',device_map=device_map,)#.to(device)
    else:base = AutoModelForCausalLM.from_pretrained(cfg['model_name'],
                                               torch_dtype='auto',device_map=device_map,)#.to(device)
    
    print("model-here2--",base.hf_device_map)
    print(torch.cuda.device_count())
    

    
  
    

    if cfg["trained_lora"] :
        #lora_cfg= PeftConfig.from_pretrained(cfg['sft_lora_dir'])
        
        model = PeftModel.from_pretrained(                      # load SFT LoRA weights
        base, cfg['sft_lora_dir'], is_trainable=True
        )#.to(device)

    else:
        lora_cfg = LoraConfig(task_type='CAUSAL_LM',
                          r=cfg['lora']['r'],
                          lora_alpha=cfg['lora']['alpha'],
                          lora_dropout=cfg['lora']['dropout'],
                          target_modules=['q_proj','k_proj','v_proj',
                                          'o_proj','gate_proj','up_proj','down_proj'])
        model = get_peft_model(base, lora_cfg)#.to(device)
    del base
    model.print_trainable_parameters()
    print("model-here--",model.hf_device_map)

     # ---- Trainer ------------------------------------------------------
    #if cfg["ReRank_batch"]:
    #    reRanker_class(batches=cfg["ReRank_batch"]).rank
    #reRanker=None#reRanker_class(batches=cfg["ReRank_batch"]).rank
    
    # ---- Trainer ------------------------------------------------------
    grpo_cfg = GRPOConfig(
        output_dir=cfg['grpo']['output_dir'],
        learning_rate=cfg['grpo']['learning_rate'], ## 1e-5,#5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        num_train_epochs=cfg['grpo']['epochs'],
        remove_unused_columns=False,
        gradient_accumulation_steps= cfg['grpo']['gradient_accumulation_steps'],
        max_completion_length=cfg['grpo']['max_completion_length'],
        num_generations=cfg['grpo']['num_generations'],
        max_prompt_length=cfg['grpo']['max_prompt_length'],
        logging_steps=cfg['grpo']['logging_steps'],
        save_strategy='steps',
        report_to=["tensorboard"],
        save_steps=cfg['grpo']['save_steps'],
        top_k=cfg['grpo']["top_k"],
        top_p=cfg['grpo']["top_p"],
        temperature=cfg['grpo']["temperature"],
        repetition_penalty=cfg['grpo']["repetition_penalty"],
        per_device_train_batch_size=cfg['grpo']["batch_size"],
        seed=42,
        label_names=[],
        beta=cfg['grpo']['kl_beta'],

    )
    
    base_reward = BaseReward(
        ce_tsv_path=cfg['reward']["ce_tsv_path"],
        top_k=cfg['reward']["top_k"], 
        tau=cfg['reward']["tau"],
        threads=30,
        debug_name=cfg['grpo']['output_dir'].split("/")[-1],
        add_init=cfg['reward']["add_init"],
        qrels=qrels,
        log_discounts_cap =10000,# cfg['reward']["top_k"],
        searcher=searcher,
        # log_dir="logs"
        )

    def wrapped_reward(completions, queries_id, **_):
        

        return base_reward(completions, queries_id,
                        searcher=None,#searcher
                        reader=None,#reader,
                        #qrels=qrels,
                        queries=queries,
                        #reRanker=reRanker,
                        DEBUG_QID=DEBUG_QID,
                        #corpus=corpus,
                        #CrossEn=ce_qrel,
                        )

    print("Training ...")

    # convert your reward_func & dataset mapping here
    trainer = GRPOTrainer(model=model,
                          args=grpo_cfg,
                          reward_funcs=wrapped_reward,
                          train_dataset=train_dataset,
                          processing_class=processing_class
                          )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(os.path.join(cfg['grpo']['output_dir'], 'last'))
    


if __name__ == '__main__':
    main()

