import glob
import networkx as nx
from karateclub.estimator import Estimator
from SG2V.signedWL import *
import pickle as pkl




"""
Transforms graph labels to be a continuous list of integers starting from 0.

:param graph: The graph to relabel.
:return: The relabeled graph.
"""
def relabel_graph(graph):
    mapping = {}
    cpt_nodes = 0
    for node in graph.nodes():
        mapping[node] = cpt_nodes
        cpt_nodes += 1
    graph = nx.relabel_nodes(graph, mapping)
    return graph


"""
Extracts the Signed Graph2Vec features (rooted subgraphs).

:param graphs_path: path of the folder containing all graph files.
:param model_type: type of model that we want to use for learning (``g2v`` or ``sg2vn`` or
        ``sg2vsb``)
:param wl_iterations: Number of iterations of the WL relabeling algorithm.
:return graphs_features: A list of extracted features.
"""
def get_graphs_features(graphs_path, model_type, wl_iterations):
    graph_files = glob.glob("%s/*.graphml" % (graphs_path))
    graphs_features = []
    for i in range(len(graph_files)):
        graph_file = "%s/%s.graphml" % (graphs_path, i)
        G = nx.read_graphml(graph_file)
        G = relabel_graph(G)
        if model_type == "g2v":
            wl_model = WeisfeilerLehmanHashing_g2v(G, wl_iterations, False, False)
        elif model_type == "sg2vn":
            wl_model = WeisfeilerLehmanHashing_sg2vn(G, wl_iterations, False, False)
        elif model_type == "sg2vsb":
            wl_model = WeisfeilerLehmanHashing_sg2vsb(G, wl_iterations, False, False)
        else:
            print ("Unknown type of model")
            break
        document = TaggedDocument(words=wl_model.get_graph_features(), tags=[str(i)])
        graphs_features.append(document)
    return graphs_features


"""
Learns the Signed Graph2Vec embeddings.

:param graph_features: The list of extracted graph features for all graphs in the dataset.
:return: List of learned embeddings.
"""
def learn_embeddings(graph_features):
    model = Signed_Graph2Vec()
    model.fit_documents(graph_features)
    embeddings = model.get_embedding()
    return embeddings

"""
Writes embeddings to local files.

:param embeddings: learned embeddings.
:param model_type: type of model used for learning (``g2v`` or ``sg2vn`` or
        ``sg2vsb``)
"""
def write_embeddings(embeddings, model_type):
    for i in range(len(embeddings)):
        with open('out/SG2V/%s/%s.pkl' % (model_type, i), 'wb') as outp:
            pkl.dump(embeddings[i], outp, pkl.HIGHEST_PROTOCOL)


"""
Runs experiments for all 3 datasets and 3 methods.
"""
def run_all_experiments_sg2v():
    for graph_path in ["data/CCS", "data/EPF", "data/SSO"]:
        for model_type in ["g2v", "sg2vn", "sg2vsb"]:
            for wl_iterations in range(5):
                graph_features  = get_graphs_features(graph_path, model_type, wl_iterations+1)
                learned_embeddings = learn_embeddings(graph_features)
                write_embeddings(learned_embeddings, model_type)


if __name__ == '__main__':
    graphs_path = "data/CCS"
    model_type = "sg2vn"
    wl_iterations = 2
    graph_features  = get_graphs_features(graphs_path, model_type, wl_iterations)
    learned_embeddings = learn_embeddings(graph_features)
    write_embeddings(learned_embeddings)