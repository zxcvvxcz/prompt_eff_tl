from transformers import AutoConfig, AutoTokenizer
from transformers.deepspeed import HfDeepSpeedConfig
import argparse
import torch
from model_wrapper.GPT2Wrapper import GPT2Wrapper
# import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
parser.add_argument(
    '--local_rank', 
    default=0, 
    type=int, 
    help='node rank for distributed training'
)
# model = AutoModel.from_pretrained("gpt2-medium")
args = parser.parse_args()
ds_config = 'ds_configs_samples/zero3_config.json'
model_name = 'EleutherAI/gpt-j-6B'
cache_dir = '/home/pch330/data/model_data'
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

config = AutoConfig.from_pretrained(model_name, num_labels=151, 
        finetuning_task='clinc150', pad_token_id=tokenizer.unk_token_id,
        apply_lora=False, lora_alpha=16, lora_r=8,
        apply_prefix=False, num_prefix=10, mid_dim=16,
        apply_encoder=False, apply_input=False, encoder_model_name_or_path='gpt2',
        freeze_encoder=False, prompt_length=None, cache_dir=cache_dir
    )

# model = GPT2Wrapper(config=config, model_name_or_path='gpt2-medium', get_last_hidden_state=True)
model = GPT2Wrapper(config=config, model_name_or_path=model_name, cache_dir=cache_dir)

# for p in model.parameters():
#     print(p.shape)



estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=8, num_nodes=1)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
