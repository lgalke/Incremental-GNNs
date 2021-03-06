import itertools as it
import os.path as osp

import dgl
from dgl import DGLGraph, DGLError
from dgl.data.utils import load_graphs
import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Globals
CACHE_DIR = '/tmp/70companies'
MEMORY = Memory(CACHE_DIR, verbose=2)

def load_data(path):
    try:
        print("Trying to load dgl graph directly")
        glist, __ = load_graphs(osp.join(path, 'g.bin'))
        g = glist[0]
        print("Success")
    except DGLError as e:
        print("File not found", e)
        print("Loading nx graph")
        nx_graph = nx.read_adjlist(osp.join(path, 'adjlist.txt'), nodetype=int)
        print("Type:", type(nx_graph))
        g = DGLGraph.from_networkx(nx_graph)
    X = np.load(osp.join(path, 'X.npy'))
    y = np.load(osp.join(path, 'y.npy'))
    t = np.load(osp.join(path, 't.npy'))
    N = g.number_of_nodes()
    assert X.shape[0] == N
    assert y.size == N
    assert t.size == N
    return g, X, y, t

def load_70companies_dataframe(path, limit=None):
    """ Loads the 70companies dataset """
    df = pd.read_csv(path, nrows=limit)
    return df

@MEMORY.cache
def load_70companies_nxgraph(path, with_features=True, vocab_size=None, limit=None):
    df = load_70companies_dataframe(path, limit=limit)
    df.reset_index(inplace=True)

    print("Creating graph")
    g = nx.Graph()
    g.add_nodes_from(range(len(df)))
    for journal, group in df.groupby('issn', sort=False):
        grp_paper_ids = group.index.values
        print(len(grp_paper_ids), "papers in", journal)
        for current in grp_paper_ids:
            # For each paper in group
            # add edges to all other papers
            g.add_edges_from((current, other) for other in grp_paper_ids)

    print("70 Companies dataset:")
    print("\tNum nodes:", g.number_of_nodes())
    print("\tNum edges:", g.number_of_edges())


    company2label = {company:i for i, company in enumerate(df.company.unique())}
    labels = [company2label[c] for c in df.company.values]
    print("\tNum labels:", len(company2label))


    if not with_features:
        return g, labels, df.year.values, len(company2label)

    tfidf = TfidfVectorizer(stop_words='english', max_features=vocab_size)
    features = tfidf.fit_transform(df.title.values)
    print("\tNum feats:", features.shape[1])

    return g, features, labels, df.year.values, len(company2label)



@MEMORY.cache
def load_70companies_dglgraph(csvfile, vocab_size=None, limit=None):
    df = load_70companies_dataframe(csvfile, limit=limit)
    graph = dgl.DGLGraph()

    years = torch.LongTensor(df.year.values)

    tfidf = TfidfVectorizer(stop_words='english', max_features=vocab_size)
    features = tfidf.fit_transform(df.title.values)

    company2label = {company:i for i, company in enumerate(df.company.unique())}
    labels = torch.LongTensor([company2label[c] for c in df.company.values])

    # nodes
    graph.add_nodes(len(df))
    # edges
    for journal, group in df.groupby('issn', sort=False):
        grp_paper_ids = group.index.values
        print(len(grp_paper_ids), "papers in", journal)
        for p in grp_paper_ids:
            # For each paper in group
            # add edges to all other papers
            graph.add_edges(p, grp_paper_ids)

    print("70 Companies dataset:")
    print("\tNum nodes:", graph.number_of_nodes())
    print("\tNum edges:", graph.number_of_edges())
    print("\tNum feats:", features.shape[1])
    print("\tNum labels:", len(company2label))

    return graph, features, labels, years, len(company2label)
