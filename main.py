import json
import pickle
import heapq
from knowledge_extractor import KnowledgeExtractor
from bert_embeddor import BERTEmbeddor
from graph_constructor import GraphConstructor
from zeroshot_classifier import ZeroshotClassifier

# Parameters
PAPER_NUM = 30
THRESHOLD = 0.8
CATEGORIES = {
    'Background': 'brief introduction to the motivation and point of departure',
    'Objective': 'what is expected to achieve by the study. It can be a survey or a review for a specific research topic, a significant scientific or engineering problem, or a demonstration for research theories or principles.',
    'Solution': 'presents the methods, models, or technologies employed in the research to achieve the research objectives',
    'Finding': 'a summary of the results'
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
with open('cache/classification_result.pickle', 'rb') as file:
    classification_result = pickle.load(file)
# classifier = ZeroshotClassifier(categories=CATEGORIES, abstracts=abstract_data)
# classification_result = classifier.classification()
# with open('cache/classification_result.pickle', 'wb') as file:
#     pickle.dump(classification_result, file)
print('Classification is done!')

# Extract Knowledge Elements and its location
extractor = KnowledgeExtractor()
knowledge_elements = extractor.extract_knowledge_elements(abstract_data)
print('Knowledge Element Extraction is done!')

# Add categories for each knowledge_elements are tagged
for element in knowledge_elements:
    categories = []
    for location in element['locations']:
        categories.append(classification_result[location[0]][location[1]])
    element['category'] = categories

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
layered_graph = {}
for category in CATEGORIES.keys():
    # Filtering nodes that have current 'category' in their 'category' attribute
    filtered_nodes = [node for node, attrs in graph.nodes(data=True) if category in attrs.get('category', [])]
    # Creating a subgraph with the filtered nodes
    layered_graph[category] = graph.subgraph(filtered_nodes)

# graph query
def query(keyword, target_network, top_k=7):
    # Find nodes with same_paper=True and target_network in category
    related_nodes = []
    for neighbor in graph.neighbors(keyword):
        edge_data = graph.get_edge_data(keyword, neighbor)
        node_data = graph.nodes[neighbor]

        if edge_data.get('same_paper', False) and target_network in node_data.get('category', []):
            related_nodes.append(neighbor)

    # Find top_k nodes with highest weights in the target network
    top_k_nodes = heapq.nlargest(top_k, related_nodes, key=lambda x: layered_graph[target_network].get_edge_data(keyword, x, {}).get('weight', 0))
    return top_k_nodes + related_nodes

def find_papers(keyword):
    locations = graph.nodes[keyword]['locations']
    titles = []
    for location in locations:
        title = data[location[0]]['title']
        if title not in titles:
            titles.append(title)
    return titles

# interactive query
while True:
    print('\n\nChoose one of these categories as source network:', CATEGORIES.keys())
    source_network = input('Source: ')
    while source_network not in CATEGORIES.keys():
        print('There is no such category')
        source_network = input('Source: ')
    
    print('Choose one of these categories as target network:', CATEGORIES.keys())
    target_network = input('Target: ')
    while target_network not in CATEGORIES.keys():
        print('There is no such category')
        target_network = input('Target: ')

    keyword_list = list(layered_graph[source_network].nodes())
    keyword_list.sort()
    print(keyword_list)
    print(f'These are keyword from {source_network} Network. Choose one of these keywords')
    keyword = input('Keyword: ')
    while keyword not in keyword_list:
        print('Thre is no such keyword')
        keyword = input('Keyword: ')
    
    results = query(keyword, target_network)
    print(f'Most related keywords in {target_network} Network.')
    print('----------------------------------------')
    print(results)
    print('----------------------------------------')
    print('Do you want to see the title of referred papers?')
    agree = input('Yes / No: ')
    if agree:
        print('\n\n')
        for result in results:
            print('----------------------------------------')
            print(f'<{result}>')
            paper_titles = find_papers(result)
            for title in paper_titles:
                print('-', title)