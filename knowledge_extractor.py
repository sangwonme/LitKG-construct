import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TextProcessor:
    def __init__(self):
        # Load the SpaCy model
        self.nlp = spacy.load("en_core_web_sm")

    # Function to extract noun phrases with their locations
    def extract_noun_phrases_with_locations(self, abstracts):
        noun_phrases = defaultdict(list)
        for paper_index, abstract in enumerate(abstracts):
            doc = self.nlp(abstract)
            for sent_index, sent in enumerate(doc.sents):
                for np in sent.noun_chunks:
                    start = np.start - sent.start
                    end = np.end - sent.start
                    noun_phrases[np.text.lower()].append((paper_index, sent_index, start, end))
        return noun_phrases

class TFIDFManager:
    def __init__(self):
        # Include up to 3-word phrases
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))

    def compute_tfidf(self, docs):
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        feature_names = self.vectorizer.get_feature_names_out()
        return tfidf_matrix, feature_names

    def filter_noun_phrases(self, noun_phrases, tfidf_matrix):
        max_tfidf_scores = defaultdict(float)
        feature_names = self.vectorizer.get_feature_names_out()

        # Create a set of all n-grams in the feature names
        feature_set = set(feature_names)

        for doc in range(tfidf_matrix.shape[0]):
            for word_idx in tfidf_matrix[doc, :].nonzero()[1]:
                word = feature_names[word_idx]
                max_tfidf_scores[word] = max(max_tfidf_scores[word], tfidf_matrix[doc, word_idx])

        # Filter noun phrases based on tf-idf score and presence in feature set
        filtered_noun_phrases = {}
        for np, locations in noun_phrases.items():
            # Normalize the noun phrase for comparison with TF-IDF features
            normalized_np = ' '.join(np.split())
            if normalized_np in feature_set and max_tfidf_scores[normalized_np] > 0.0:
                filtered_noun_phrases[np] = locations

        return filtered_noun_phrases


class KnowledgeExtractor:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.tfidf_manager = TFIDFManager()

    # Function to extract and filter knowledge elements
    def extract_knowledge_elements(self, abstracts):
        noun_phrases = self.text_processor.extract_noun_phrases_with_locations(abstracts)
        tfidf_matrix, _ = self.tfidf_manager.compute_tfidf(abstracts)
        filtered_noun_phrases = self.tfidf_manager.filter_noun_phrases(noun_phrases, tfidf_matrix)

        knowledge_elements = []
        for np, locations in filtered_noun_phrases.items():
            knowledge_elements.append({
                'keyword': np,
                'locations': locations
            })
        return knowledge_elements
