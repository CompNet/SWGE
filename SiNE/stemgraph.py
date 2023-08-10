import networkx as nx
import numpy as np

# From https://github.com/CompNet/SignedCentrality

def get_empty_graph(directed=True):
    if directed:
        return nx.DiGraph()
    return nx.Graph()

def from_edgelist_array_to_graph(X, y, directed=True):
    positive_graph = get_empty_graph(directed)
    negative_graph = get_empty_graph(directed)

    for edge, label in zip(X, y):
        u, v = edge
        if label == 0:
            negative_graph.add_edge(u, v)
        else:
            positive_graph.add_edge(u, v)
    return positive_graph, negative_graph


def get_triples(positive_graph, negative_graph, p0=True):
    triples = []
    triples0 = []
    for u, v in positive_graph.edges():
        if v in negative_graph:
            v_neigbors = negative_graph[v]
            for w in v_neigbors:
                triple = (u, v, w)
                triples.append(triple)
        elif p0:
            triple0 = (u, v, 0)
            triples0.append(triple0)
    triples = np.array(triples)
    triples0 = np.array(triples0)
    if p0:
        return triples, triples0
    return triples


