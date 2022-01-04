
from typing import Tuple

import torch

from transformers import AutoModel, T5EncoderModel, T5ForConditionalGeneration
from deepspeed.runtime.utils import see_memory_usage

from .InputProcessor import *
from .OutputProcessor import BaseOutputProcessor, T5OutputProcessor



# class T5Wrapper(torch.nn.Module):
#     #! raw
#     # def __init__(self, config, model_name_or_path):
#     #! raw
    
#     #! HS
#     def __init__(self, config, model_name_or_path, tokenizer=tokenizer, max_label_length = max_label_length):
#     #! HS
#         super(T5Wrapper, self).__init__()

#         self.config = config

#         # Main model
#         self.transformer = T5ForConditionalGeneration.from_pretrained(
#                                         model_name_or_path,
#                                         from_tf=bool(".ckpt" in model_name_or_path),
#                                         config=config)
        
#         #! HS
#         self.tokenizer = tokenizer
#         self.max_label_length = max_label_length
#         #! HS

#         self.embedding_dim = self.transformer.get_input_embeddings().embedding_dim
#         self.num_labels = config.num_labels

#         # for output processing (output logits -> loss, prediction)
#         self.output_processor = BaseOutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)

#         # for other methods (LoRA, Adapter, Prefix-tuning)
#         # input_ids -> input_embeds
#         if not self.config.apply_input and not self.config.apply_encoder and self.config.prompt_length is None:
#             self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
#         # for PROMPT_TUNING
#         elif not self.config.apply_input and not self.config.apply_encoder:
#             self.input_processor = PromptInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
#         # for PLM encoder + prompt only
#         elif not self.config.apply_input and self.config.apply_encoder:
#             self.input_processor = PromptEncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
#         # for PLM encoder + input dependent
#         elif self.config.apply_input and self.config.apply_encoder:
#             self.input_processor = EncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        

#     def forward(
#         self,
#         input_ids=None,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:

#         # inputs_embeds  : (batch, input_length, embedding_dim)
#         # attention_mask : (batch, input_length)
#         inputs_embeds, attention_mask = self.input_processor(input_ids=input_ids, attention_mask=attention_mask)

#         #! raw
#         # outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
#         # shape : (batch, length, embedding_dim)
#         # last_hidden_state = outputs.last_hidden_state
#         # 
#         # loss        : (batch, )
#         # predictions : (batch, )
#         # loss, predictions = self.output_processor(last_hidden_state=last_hidden_state, attention_mask=attention_mask, labels=labels)
#         # 
#         # return loss, predictions
#         #! raw

#         #! HS
#         outputs = self.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss
        
#         dec_seq_rep = outs.decoder_hidden_states
#         last_seq_rep = dec_seq_rep[-1][:,:self.max_label_length,:]
#         last_hid_rep = out_seq_rep.sum(1)

#         # outs = self.transformer.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_length = self.max_label_length)
#         # prediction_list = self.tokenizer.batch_decode(outs)
#         # prediction_list = [self.tokenizer.decode(ids) for ids in outs]

#         # raw -> (batch,1) Tensor
#         # prediction_list -> list (batch, seq_length)      where,  max seq_length = label label length

#         return loss, prediction_list
#         #! HS
        
class T5EncWrapper(torch.nn.Module):
    def __init__(self, config, model_name_or_path, cache_dir=None, get_last_hidden_state=False, load_init_model=False, ckpt_path=None):
        super(T5EncWrapper, self).__init__()

        self.config = config
        
        self.get_last_hidden_state = get_last_hidden_state
        # Main model
        see_memory_usage('No model initialized', True)
        if load_init_model:
            self.transformer = T5EncoderModel.from_config(config)
        else:
            self.transformer = T5EncoderModel.from_pretrained(
                                            model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=config, cache_dir=cache_dir)

        self.embedding_dim = self.transformer.get_input_embeddings().embedding_dim
        self.num_labels = config.num_labels
        see_memory_usage('Transformer initialized', True)

        # for output processing (output logits -> loss, prediction)
        self.output_processor = T5OutputProcessor(config=config, embedding_dim=self.embedding_dim, num_labels=self.num_labels)

        see_memory_usage('output_processor initialized', True)
        # for other methods (LoRA, Adapter, Prefix-tuning)
        # input_ids -> input_embeds
        if not self.config.apply_input and not self.config.apply_encoder and self.config.prompt_length is None:
            print('get base input processor')
            self.input_processor = BaseInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PROMPT_TUNING
        elif not self.config.apply_input and not self.config.apply_encoder:
            self.input_processor = PromptInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PLM encoder + prompt only
        elif not self.config.apply_input and self.config.apply_encoder:
            self.input_processor = PromptEncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())
        # for PLM encoder + input dependent
        elif self.config.apply_input and self.config.apply_encoder:
            self.input_processor = EncoderInputProcessor(config=config, embeddings=self.transformer.get_input_embeddings())

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
            