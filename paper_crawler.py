import json
import requests

class PaperCrawler:
    def __init__(self, base_url="http://api.semanticscholar.org/graph/v1"):
        self.base_url = base_url

    def search_papers(self, query, fields, year_range="2019-", output_file="./data/papers.jsonl"):
        if isinstance(query, list):
            for q in query:
                self._search_single_query(q, fields, year_range, output_file)
        else:
            self._search_single_query(query, fields, year_range, output_file)

    def _search_single_query(self, query, fields, year_range, output_file):
        url = f"{self.base_url}/paper/search/bulk?query={query}&fields={fields}&year={year_range}"
        response = requests.get(url).json()

        print(f"Will retrieve an estimated {response['total']} documents for query: {query}")
        retrieved = 0

        with open(output_file, "w") as file:
            while True:
                if "data" in response:
                    retrieved += len(response["data"])
                    print(f"Retrieved {retrieved} papers for query: {query}...")
                    for paper in response["data"]:
                        print(json.dumps(paper), file=file)
                if "token" not in response:
                    break
                response = requests.get(f"{url}&token={response['token']}").json()

        print(f"Done! Retrieved {retrieved} papers total for query: {query}")

    def read_paper_ids(self, file_path):
        paper_ids = []
        with open(file_path, 'r') as file:
            for line in file:
                paper = json.loads(line)
                paper_id = paper.get('paperId')
                if paper_id:
                    paper_ids.append(paper_id)
        return paper_ids

    def retrieve_paper_details(self, paper_ids, output_file="./data/data.json"):
        data = []
        for i, paper_id in enumerate(paper_ids):
            url = f"{self.base_url}/paper/{paper_id}?fields=url,year,title,authors,venue,abstract,references,citations"
            response = requests.get(url).json()
            data.append(response)
            print(f"Retrieving: {i} - ID: {paper_id}")

        with open(output_file, 'w') as file:
            json.dump(data, file)

    def read_paper_ids(self, file_path):
        paper_ids = []
        with open(file_path, 'r') as file:
            for line in file:
                paper = json.loads(line)
                paper_id = paper.get('paperId')
                if paper_id:
                    paper_ids.append(paper_id)
        return paper_ids

    def retrieve_paper_details(self, paper_ids, output_file="./data/data.json"):
        data = []
        for i, paper_id in enumerate(paper_ids):
            url = f"{self.base_url}/paper/{paper_id}?fields=url,year,title,authors,venue,abstract,references,citations"
            response = requests.get(url).json()
            data.append(response)
            print(f"Retrieving: {i} - ID: {paper_id}")

        with open(output_file, 'w') as file:
            json.dump(data, file)