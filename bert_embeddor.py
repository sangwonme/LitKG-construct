import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

class BERTEmbeddor:
    def __init__(self, model_name='bert-base-uncased', cache_dir='cache'):
        self.cache_dir = cache_dir
        self.model_dir = os.path.join(cache_dir, model_name)
        self.load_model_and_tokenizer(model_name)

    def load_model_and_tokenizer(self, model_name):
        if not os.path.exists(self.model_dir):
            print("Downloading BERT model and tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            # Saving using their native save methods
            self.tokenizer.save_pretrained(self.model_dir)
            self.model.save_pretrained(self.model_dir)
        else:
            print("Loading BERT model and tokenizer from cache...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
            self.model = BertModel.from_pretrained(self.model_dir)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state
