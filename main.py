import json
from knowledge_extractor import KnowledgeExtractor
from bert_embeddor import BERTEmbeddor
from graph_constructor import GraphConstructor

# Parameters
PAPER_NUM = 30

# File Path
DATA_PATH = './data/data1.json'

# Load Paper data
with open(DATA_PATH, encoding='utf-8') as file:
    data = json.load(file)

# Concat All Abstracts
abstract_data = [data[i]['abstract'] if data[i]['abstract'] else '' for i in range(len(data))]

# Extract Knowledge Elements and its location
extractor = KnowledgeExtractor()
knowledge_elements = extractor.extract_knowledge_elements(abstract_data[:PAPER_NUM])
print('Knowledge Element Extraction is done!')

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

import pdb; pdb.set_trace()