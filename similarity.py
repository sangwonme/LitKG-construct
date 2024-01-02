from transformers import BertTokenizer, BertModel
import numpy as np
from scipy.spatial.distance import cosine
import pickle

class Similarity:
    def __init__(self):
        # Load BERT model and tokenizer
        model_file = './cache/model.pkl'
        tokenizer_file = './cache/tokenizer.pkl'
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        except:
            self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
            with open(model_file, 'wb') as f:
                pickle.dump(self.model, f)
        try:
            with open(tokenizer_file, 'rb') as f:
                self.tokenizer = pickle.load(f)
        except:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            with open(tokenizer_file, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            
            
    # Function to get BERT embedding
    def get_bert_embedding(self, sentence):
        # Tokenize the sentence. The tokenizer returns PyTorch tensors by default.
        inputs = self.tokenizer(sentence, return_tensors="pt")

        # Pass the tokenized inputs to the model. No need for 'return_tensors' here.
        outputs = self.model(**inputs)

        # Detach the output tensor from the computational graph and convert to NumPy array
        return outputs.last_hidden_state.mean(1).squeeze().detach().numpy()
    
    # Function to calculate cosine similarity between all pairs
    def calculate_similarity(self, text_list):
        embeddings = [self.get_bert_embedding(text) for text in text_list]
        num_texts = len(embeddings)
        similarity_matrix = np.zeros((num_texts, num_texts))

        for i in range(num_texts):
            for j in range(num_texts):
                if i != j:
                    similarity_matrix[i][j] = 1 - cosine(embeddings[i], embeddings[j])
                else:
                    similarity_matrix[i][j] = 0
        return similarity_matrix