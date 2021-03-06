
from typing import Tuple
import pdb
import torch

from transformers import AutoModel
from deepspeed.runtime.utils import see_memory_usage

from .InputProcessor import *
from .OutputProcessor import BaseOutputProcessor



class GPT2Wrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path, cache_dir=None, get_last_hidden_state=False, load_init_model=False, ckpt_path=None):
        super(GPT2Wrapper, self).__init__()

        self.config = config
        
        self.get_last_hidden_state = get_last_hidden_state
        # Main model
        see_memory_usage('No model initialized', True)
        if load_init_model:
            self.transformer = AutoModel.from_config(config)
        else:
            self.transformer = AutoModel.from_pretrained(
                                            model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=config, cache_dir=cache_dir)

        self.embedding_dim = self.transformer.wte.embedding_dim
        self.num_labels = config.num_labels
        see_memory_usage('Transformer initialized', True)

        # for output processing (output logits -> loss, prediction)
        self.output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)

        see_memory_usage('output_processor initialized', True)
        # for other methods (LoRA, Adapter, Prefix-tuning)
        # input_ids -> input_embeds
        if not self.config.apply_input and not self.config.apply_encoder and self.config.prompt_length is None:
            print('get base input processor')
            self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.wte)
        # for PROMPT_TUNING
        elif not self.config.apply_input and not self.config.apply_encoder:
            self.input_processor = PromptInputProcessor(config=config, embeddings=self.transformer.wte)
        # for PLM encoder + prompt only
        elif not self.config.apply_input and self.config.apply_encoder:
            self.input_processor = PromptEncoderInputProcessor(config=config, embeddings=self.transformer.wte)
        # for PLM encoder + input dependent
        elif self.config.apply_input and self.config.apply_encoder:
            self.input_processor = EncoderInputProcessor(config=config, embeddings=self.transformer.wte)

        see_memory_usage('input_processor initialized', True)
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # inputs_embeds  : (batch, input_length, embedding_dim)
        # attention_mask : (batch, input_length)
        inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, attention_mask=attention_mask)

        outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # shape : (batch, length, embedding_dim)
        last_hidden_state = outputs.last_hidden_state

        # loss        : (batch, )
        # predictions : (batch, )
        # logits      : (batch, num_labels)
        loss, logits = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)
        
        result = (loss, logits, last_hidden_state) if self.get_last_hidden_state else (loss, logits)
        return result
            
