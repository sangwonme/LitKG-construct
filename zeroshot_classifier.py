# input: categories and its definition
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import ast
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ZeroshotClassifier:
    def __init__(self, categories, abstracts):
        # Load the SpaCy model
        self.nlp = spacy.load("en_core_web_sm")
        # Set up the turbo LLM
        with open('api_key.txt','r') as file:
            api_key = file.read()
        self.llm = ChatOpenAI(
            temperature=0,
            model_name='gpt-3.5-turbo',
            openai_api_key=api_key
        )
        print('LLM setup is done!')
        # abstract data with splitted sentences
        self.abstracts = []
        self.split_sentences_abstract(abstracts)
        print('Splitting abstract sentences is done!')
        # categories output
        self.categories = categories

    # Function to extract noun phrases with their locations
    def split_sentences_abstract(self, abstracts):
        for paper_index, abstract in enumerate(abstracts):
            doc = self.nlp(abstract)
            sentences = []
            for sent_index, sent in enumerate(doc.sents):
                sentences.append('\"'+str(sent)+'\"')
            self.abstracts.append(sentences)
            print('Abstract', paper_index, 'is splitted.')
    
    # Classifier
    def classification(self):
        classification_result = []
        for paper_index, abstract in enumerate(self.abstracts):
            messages = [
                SystemMessage(
                    content="You are a helpful assistant that classify the sentences in given categories."
                ),
                HumanMessage(content=f"""
                I will give you abstract of research paper.
                Your role is to classify each sentence in one of following categories.
                - Background: brief introduction to the motivation and point of departure
                - Objective: what is expected to achieve by the study. It can be a survey or a review for a specific research topic, a significant scientific or engineering problem, or a demonstration for research theories or principles.
                - Solution: presents the methods, models, or technologies employed in the research to achieve the research objectives
                - Findings: a summary of the results

                Could you label all sentences in the abstract one by one please?
                - Do not give the description just following answer

                <Input Format>
                ['sentence 1', 'sentence 2', ... , 'sentence N']
                <Output Format>
                ['category for sentence 1', 'category for sentence 2', '....']

                <Abstract>
                {str(abstract)}
                """),
            ]
            print(abstract)
            print('len :', len(abstract))
            result = self.llm(messages)
            result = ast.literal_eval(result.content)
            classification_result.append(result)
            print('Paper', paper_index, 'classified.')
            
        return classification_result
