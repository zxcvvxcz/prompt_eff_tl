from transformers import BertTokenizer, BertModel
import torch
import pdb
import os
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=os.path.join(os.path.expanduser('~'), 'data', 'model_data'))
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, cache_dir=os.path.join(os.path.expanduser('~'), 'data', 'model_data'))

inputs = tokenizer("Hello, my dog is so cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
attentions = outputs.attentions
pdb.set_trace()