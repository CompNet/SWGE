import glob
import networkx as nx
from karateclub.estimator import Estimator
from signedWL import *
import pickle as pkl




"""Convert a graph ...

:param 
:return:
"""




def relabel_graph(graph):
    mapping = {}
    cpt_nodes = 0
    for node in graph.nodes():
        mapping[n] = cpt_nodes
        cpt_nodes += 1
    graph = nx.relabel_nodes(graph, mapping)
    return graph

def get_graphs_features(graphs_path, model_type, wl_iterations):
"""Load a collection of graphs.

:param graphs_path: path of the directory containing all graph files.
:param model_type: type of model that we want to use for learning (``g2v`` or ``sg2vn`` or
        ``sg2vsb``)
:return:
"""
    graph_files = glob.glob("%s/*.graphml" % (graphs_path))
    graphs_features = []
    for i in range(len(graph_files)):
        graph_file = "%s/%s.graphml" % (graphs_path, i)
        G = nx.read_grahml(graph_file)
        G = relabel_graph(G)
        G = self._check_graph(G)
        if model_type == "g2v":
            #graph, wl_iterations, attributed, erase_base_features
            wl_model = WeisfeilerLehmanHashing_g2v(G, wl_iterations, False, False)
        elif model_type == "sg2vn":
            #graph, wl_iterations, attributed, erase_base_features
            wl_model = WeisfeilerLehmanHashing_sg2vn(G, wl_iterations, False, False)
        elif model_type == "sg2vsb":
            #graph, wl_iterations, attributed, erase_base_features
            wl_model = WeisfeilerLehmanHashing_sg2vsb(G, wl_iterations, False, False)
        else:
            print ("Unknown type of model")
            break
        document = TaggedDocument(words=wl_model.get_graph_features(), tags=[str(i)])
        graphs_features.append(document)
    return graphs_features


def learn_embeddings(graph_features):
    model = Signed_Graph2Vec()
    model.fit_documents(graph_features)
    embeddings = model.get_embedding()
    return embeddings


def write_embeddings(embeddings, path):
    for i in range(len(embeddings)):
    with open('%s/%s.pkl' % (path, i), 'wb') as outp:
        pickle.dump(embeddings[i], outp, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    graph_features  = get_graphs_features(graphs_path, model_type, wl_iterations)
    learned_embeddings = learn_embeddings(graph_features)
    write_embeddings(learned_embeddings)