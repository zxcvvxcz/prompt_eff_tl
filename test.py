from transformers import AutoConfig, AutoTokenizer
import torch
from models.GPT2Wrapper import GPT2Wrapper
# model = AutoModel.from_pretrained("gpt2-medium")

tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

config = AutoConfig.from_pretrained('gpt2-medium', num_labels=151, 
        finetuning_task='clinc150', pad_token_id=tokenizer.unk_token_id,
        apply_lora=False, lora_alpha=16, lora_r=8,
        apply_prefix=False, num_prefix=10, mid_dim=16,
        apply_encoder=False, apply_input=False, encoder_model_name_or_path='gpt2',
        freeze_encoder=False, prompt_length=None, cache_dir=None
    )
# model = GPT2Wrapper(config=config, model_name_or_path='gpt2-medium', get_last_hidden_state=True)
model = GPT2Wrapper(config=config, model_name_or_path='gpt2-medium', cache_dir=None)

param = model.parameters()
for p in param:
    print(p.shape)