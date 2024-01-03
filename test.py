from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Set up the turbo LLM
llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo',
    openai_api_key='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'
)
print('setup llm -------------')

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

tmp = llm(messages)

import pdb; pdb.set_trace()