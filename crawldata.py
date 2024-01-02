import requests
import json

def read_paper_ids(file_path):
    paper_ids = []
    with open(file_path, 'r') as file:
        for line in file:
            paper = json.loads(line)  # Parse the JSON line
            paper_id = paper.get('paperId')  # Extract the paperId
            if paper_id:
                paper_ids.append(paper_id)
    return paper_ids

paper_ids = read_paper_ids('papers.jsonl')

data = []
for i, id in enumerate(paper_ids):
    url = f'https://api.semanticscholar.org/graph/v1/paper/{id}?fields=url,year,title,authors,venue,abstract,references,citations'
    request = requests.get(url).json()
    data.append(request)
    print(i, id)

with open('./data.json', 'w') as file:
    json.dump(data, file)

import pdb; pdb.set_trace()