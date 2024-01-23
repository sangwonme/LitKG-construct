import streamlit as st
from pathlib import Path
import os
import shutil
import pickle
import spacy

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text(encoding='utf-8')

def delete_files_in_path(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

st.set_page_config(
    page_title="KG Experiment",
    page_icon=":spider:",
    layout="wide",
)
st.config.set_option("server.fileWatcherType", "none")

# ===============================================================================
# utils

# Function to extract noun phrases with their locations
@st.cache_data
def split_sentences_abstract(abstracts):
    abstracts_splitted = []
    nlp = spacy.load("en_core_web_sm")
    for paper_index, abstract in enumerate(abstracts):
        doc = nlp(abstract)
        sentences = []
        for sent_index, sent in enumerate(doc.sents):
            sentences.append('\"'+str(sent)+'\"')
        abstracts_splitted.append(sentences)
        print('Abstract', paper_index, 'is splitted.')
    return abstracts_splitted


# ===============================================================================
st.title("KG Experiment")

FILE_NAME = 'seed_data'
GRAPH_PATH = f'../cache/{FILE_NAME}_graph.pickle'
LAYEREDGRAPH_PATH = f'../cache/{FILE_NAME}_layeredgraph.pickle'
ABSTRACT_PATH = f'../cache/{FILE_NAME}_abstract.pickle'

# load data
with open(GRAPH_PATH, 'rb') as file:
    graph = pickle.load(file)
with open(LAYEREDGRAPH_PATH, 'rb') as file:
    layered_graph = pickle.load(file)
with open(ABSTRACT_PATH, 'rb') as file:
    abstracts = pickle.load(file)

# abstract split the sentences
abstracts_splitted = split_sentences_abstract(abstracts)

# st.write(layered_graph['Background'].nodes())
st.write(layered_graph['Background'].nodes['love'])
# st.write(type(layered_graph['Background']))

'# Explore'
col1, col2 = st.columns([3, 5])
with col1:
    '### Step 1. Input your curiosity'
    tab1, tab2 = st.tabs(['Manual', 'GPT'])
    with tab1:
        source_domain = st.selectbox('Source Domain', layered_graph.keys())
        source_node = st.selectbox(f'Choose Node in {source_domain} Domain', layered_graph[source_domain].nodes(), index=1)
        target_domain = st.selectbox('Target Domain', layered_graph.keys())
        target_node = st.selectbox(f'Choose Node in {target_domain} Domain', layered_graph[target_domain].nodes(), index=2)

    with tab2:
        query_text = st.text_area('Explore with any curiosity!', height=120)
        # TODO: Natural language -> []
    

        send_button = st.button('Send', type='primary')
    '### Step 2. Control some parameters'
    sim_th = st.slider('Similarity Threshold', 0, 100, 50)
with col2:
    '### Graph Result'
    'Select the nodes to find out insight!'
    