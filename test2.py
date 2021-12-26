from transformers import AutoConfig, AutoTokenizer
# from transformers.deepspeed import HfDeepSpeedConfig
import torch
from model_wrapper.GPT2Wrapper import GPT2Wrapper
import deepspeed
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

# model = AutoModel.from_pretrained("gpt2-medium")
# vxzcvc = HfDeepSpeedConfig('ds_configs_samples/zero3_config.json')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B', cache_dir='model_data_gpt-j-6B')

config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B', num_labels=151, 
        finetuning_task='clinc150', pad_token_id=tokenizer.unk_token_id,
        apply_lora=False, lora_alpha=16, lora_r=8,
        apply_prefix=False, num_prefix=10, mid_dim=16,
        apply_encoder=False, apply_input=False, encoder_model_name_or_path='gpt2',
        freeze_encoder=False, prompt_length=None, cache_dir=None
    )

# model = GPT2Wrapper(config=config, model_name_or_path='gpt2-medium', get_last_hidden_state=True)
# with deepspeed.zero.Init(config_dict_or_path='ds_configs_samples/zero3_config.json'):
model = GPT2Wrapper(config=config, model_name_or_path='EleutherAI/gpt-j-6B', cache_dir='model_data_gpt-j-6B')

# for p in model.parameters():
#     print(p.shape)



estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=2, num_nodes=1)