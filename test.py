from transformers import BertModel, BertTokenizer
import torch
from torch.nn.functional import cosine_similarity

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    # Tokenize the text and convert to tensor
    inputs = tokenizer(text, return_tensors='pt')
    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the embeddings from the last hidden state
    embeddings = outputs.last_hidden_state
    # Pool the outputs into a mean vector
    mean_embedding = embeddings.mean(dim=1)
    return mean_embedding

# Texts
text1 = "Your first text here."
text2 = "Your second text here."

# Get embeddings
embedding1 = get_bert_embedding(text1)
embedding2 = get_bert_embedding(text2)

# Calculate cosine similarity
cos_sim = cosine_similarity(embedding1, embedding2)

print("Cosine Similarity:", cos_sim.item())
