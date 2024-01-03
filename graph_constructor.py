import networkx as nx
import torch
from sklearn.metrics.pairwise import cosine_similarity

class GraphConstructor:
    def __init__(self, knowledge_elements):
        self.knowledge_elements = knowledge_elements
        self.graph = nx.Graph()
        self.create_graph()

    def create_graph(self):
        # Add nodes
        for element in self.knowledge_elements:
            # Convert BERT embeddings to mean embeddings
            mean_embedding = torch.mean(element['bert'], dim=1).squeeze()
            self.graph.add_node(element['keyword'], bert=mean_embedding.numpy(), locations=element['locations'], category=element['category'])

        # Add edges with weights
        for i, element_i in enumerate(self.knowledge_elements):
            for j, element_j in enumerate(self.knowledge_elements):
                if i < j:
                    weight = self.calculate_edge_weight(element_i, element_j)
                    same_paper = any(loc_i[0] == loc_j[0] for loc_i in element_i['locations'] for loc_j in element_j['locations'])
                    self.graph.add_edge(element_i['keyword'], element_j['keyword'], weight=weight, same_paper=same_paper)

    def calculate_edge_weight(self, element_i, element_j, weight_similarity=0.8, weight_distance=0.2):
        # Cosine similarity
        cos_sim = cosine_similarity(element_i['bert'], element_j['bert']).item()

        # Distance between locations (example calculation)
        location_distance = 1
        for location_i in element_i['locations']:
            for location_j in element_j['locations']:
                # Update only when in same paper
                if location_i[0] == location_j[0]:
                    tmp = abs(location_i[1] - location_j[1]) / 8
                    location_distance = min(location_distance, tmp)
                    # TODO: I want to add edge 'same_paper'=True when location_i[0]==location_j[0] once

        # Weighted sum of cos_sim and location_distance
        # Adjust the weights as per your requirement
        weight = weight_similarity*cos_sim + weight_distance*(1-location_distance)
        return weight
