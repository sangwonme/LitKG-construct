from paper_crawler import PaperCrawler

crawler = PaperCrawler()
titles = [
      'Captivate! contextual language guidance for parent–child interaction',
      'Development and preliminary testing of the new five-level version of EQ-5D (EQ-5D-5L)'
      ]
crawler.search_papers(titles, "title,year")
paper_ids = crawler.read_paper_ids('./data/papers.jsonl')
crawler.retrieve_paper_details(paper_ids)