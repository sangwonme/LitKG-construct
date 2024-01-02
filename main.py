import json
from knowledge_extractor import KnowledgeExtractor

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

# Add BERT embedding attribute for all knowledge elements
# TODO

# Graph Construction with calculating links
# TODO