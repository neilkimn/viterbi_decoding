import streamlit as st
import codecs
import numpy as np
import myutils
from hmm import HMM
from collections import defaultdict
import pydot
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout, write_dot

# load data
train_data = myutils.read_conll_file("data/da_ddt-ud-train.conllu")
#dev_data = myutils.read_conll_file("data/da_ddt-ud-dev.conllu")

st.title('Visualisation of the Viterbi decoding algorithm for POS-tagging.')

text = st.text_input('Input text for tagging', 'Hvor kommer julemanden fra ?')

hmm = HMM()

hmm.fit(train_data)

test_text = text.split()

preds = hmm.predict(test_text, method='viterbi')

G = hmm.graph

pos=nx.drawing.nx_agraph.graphviz_layout(
        G,
        prog='dot',
        args='-Grankdir=LR'
    )

edges, weights = zip(*nx.get_edge_attributes(G,'weight').items())

fig = plt.figure(figsize=(15,15)) 

nx.draw(
    G,
    node_size=2000,
    node_color='#8f8f8f',
    edgelist=edges,
    edge_color=weights,
    width=10,
    edge_cmap=plt.cm.Blues,
    arrowsize=1,
    with_labels=True,
    labels={n: n for n in G.nodes},
    font_color='#FFFFFF',
    font_size=12,
    pos=pos
)

labels = nx.get_edge_attributes(G,'weight')

nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

tags = ' '.join(preds[0])
st.write("POS-tags for input sentence: ", tags)

st.pyplot(fig)