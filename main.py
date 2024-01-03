import json
from knowledge_extractor import KnowledgeExtractor
from bert_embeddor import BERTEmbeddor
from graph_constructor import GraphConstructor
from zeroshot_classifier import ZeroshotClassifier

# Parameters
PAPER_NUM = 30
THRESHOLD = 0.8
CATEGORIES = {
    'background': 'brief introduction to the motivation and point of departure',
    'objective': 'what is expected to achieve by the study. It can be a survey or a review for a specific research topic, a significant scientific or engineering problem, or a demonstration for research theories or principles.',
    'solution': 'presents the methods, models, or technologies employed in the research to achieve the research objectives',
    'finding': 'a summary of the results'
}

# File Path
DATA_PATH = './data/data1.json'

# Load Paper data
with open(DATA_PATH, encoding='utf-8') as file:
    data = json.load(file)
data = data[:PAPER_NUM]

# Concat All Abstracts
abstract_data = [data[i]['abstract'] if data[i]['abstract'] else '' for i in range(len(data))]

# Classifying categories for each sentences
classifier = ZeroshotClassifier(categories=CATEGORIES, abstracts=abstract_data)
classification_result = classifier.classification()
print('Classification is done!')

# Extract Knowledge Elements and its location
extractor = KnowledgeExtractor()
knowledge_elements = extractor.extract_knowledge_elements(abstract_data)
print('Knowledge Element Extraction is done!')

# Add categories for each knowledge_elements are tagged
# TODO

# Add BERT embedding attribute for all knowledge elements
embeddor = BERTEmbeddor()
for element in knowledge_elements:
    element['bert'] = embeddor.get_embedding(element['keyword'])
print('BERT Embedding is done!')

# Graph Construction with calculating links
constructor = GraphConstructor(knowledge_elements)
graph = constructor.graph
print('Graph Construction is done!')
print('---------------------------------')
print(f'Nodes: {len(graph.nodes)}')
print(f'Edges: {len(graph.edges)}')
print('---------------------------------')

# Split the graph regarding categories
# TODO

import pdb; pdb.set_trace()