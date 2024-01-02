import spacy
import networkx as nx
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load models
nlp = spacy.load("en_core_web_sm")
model_file = './cache/model.pkl'
tokenizer_file = './cache/tokenizer.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)
with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)

# Load Abstract data
# 3D List: Papers -> Abstract Sentences -> Words
# TODO

# Concat All Abstracts
# TODO


# Extract Knowledge Elements and its location
# TODO


# Ed
