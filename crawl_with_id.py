import requests
import json

FILE_NAME = 'seed'
DATA_PATH = f'./data/{FILE_NAME}.json'

# Load Paper data
with open(DATA_PATH, encoding='utf-8') as file:
    data = json.load(file)

ids = [data[i]['paperId'] for i in range(len(data))]

r = requests.post(
    'https://api.semanticscholar.org/graph/v1/paper/batch',
    params={'fields': 'url,year,title,authors,venue,abstract,references,citations'},
    json={"ids": ids}
)
# print(json.dumps(r.json(), indent=2))

with open('./data/seed_data.json', 'w') as file:
    json.dump(r.json(), file)

import pdb; pdb.set_trace()