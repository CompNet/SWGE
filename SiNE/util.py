import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
import SiNEmaster.stemgraph as graph  # Altered code.
import networkx as nx

# From https://github.com/CompNet/SignedCentrality

def get_num_unbalanced(X, y):
    indicies = np.arange(0, len(X))
    count = 0.0
    negatives = 0.0
    for u, v in X:
        connecting_idx = indicies[X[:,0] == v]
        count += len(connecting_idx)
        negs = 1 - y[connecting_idx]
        negatives += sum(negs)
        
#         connecting_idx = indicies[X[:,1] == v]
#         count += len(connecting_idx)
#         negs = 1 - y[connecting_idx]
#         negatives += sum(negs)
        
    return negatives / count

def remove_edge_sign(input_filepath, output_filepath, indelimiter=',', outdelimiter=' '):
    with open(input_filepath, 'r') as ifp:
        with open(output_filepath, 'w') as ofp:
            for line in ifp:
                contents = line.split(indelimiter)
                ofp.write(outdelimiter.join(contents[:-1]) + '\n')

def embedding_layer_from_numpy(arr, start_with=1):
    rows, cols = arr.shape
    if start_with == 1:
        rows += 1
    embeddings = nn.Embedding(rows, cols)
    brr = arr
    if start_with == 1:
        temp = np.random.rand(1, cols)
        brr = np.concatenate((temp, arr))
    brr = nn.Parameter(brr)
    embeddings.weight = brr
    return embeddings

def embedding_from_file(filepath, ids_explicit=True, delimiter=' ', start_with=1):
    with open(filepath, 'r') as fp:
        count = start_with
        nrows = 0
        d = dict()
        for line in fp:
            data = map(float, line.split(delimiter))
            if ids_explicit:
                d[int(data[0])] = data[1:]
                nrows = max(nrows, int(data[0]))
            else:
                d[count] = data
                nrows = count
                count += 1
        sorted_keys = sorted(d.keys())
        arr = []
        for key in sorted_keys:
            arr.append(d[key])
        arr = np.array(arr)
        return embedding_layer_from_numpy(arr, start_with)


def array_edgelist_to_graph(X, directed=False):
    graph = nx.DiGraph() if directed else nx.Graph()
    for u, v in X:
        graph.add_edge(u, v)
    return graph
    












def tensorfy_col(x, col_idx, tensor_type='long'):
    """
    Extracts a column from a numpy array and wraps it as a PyTorch Variable
    Parameters
    ----------
    x : np.array
        A 2D Numpy array
    col_idx : int
        The column to extract
    tensor_type : str (optional)
        The type of tensor to create from the column (default is 'long')

    Returns
    -------
    Pytorch Variable
        A Pytorch Variable wrapping the specified column from the submitted array

    """
    col = x[:,col_idx]
    if tensor_type == 'long':
        col = torch.LongTensor(col)
    if tensor_type == 'float':
        col = torch.FloatTensor(col)
    col = Variable(col)
    return col



def get_triples_training_batch(triples, batch_size):
    nrows = triples.shape[0]
    rows = None  # Added code.
    try:  # Added code.
         rows = np.random.choice(nrows, batch_size, replace=False)  # Altered code.
    except:  # Added code.
        rows = np.random.choice(nrows, batch_size, replace=True)  # Added code.
    choosen = triples[rows,:]
    xi = tensorfy_col(choosen, 0)
    xj = tensorfy_col(choosen, 1)
    xk = tensorfy_col(choosen, 2)
    return xi, xj, xk


def hadamard(x, y):
    return x * y


def average(x, y):
    return (x + y)/2.0


def l1(x, y):
    return np.abs(x - y)


def l2(x, y):
    return np.power(x - y, 2)


def concat(x, y):
    return np.concatenate((x, y), axis=0)


def graph_from_numpy_array(X_train, num_nodes, directed=True):
    all_nodes = set(range(0, num_nodes + 1))
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(all_nodes)
    for u, v in X_train:
        graph.add_edge(u, v)
    return graph

FEATURE_FUNCS = {
    'l1': l1,
    'l2': l2,
    'concat': concat,
    'average': average,
    'hadamard': hadamard
}


def triples_from_array(X, y, directed=True):
    p, n = graph.from_edgelist_array_to_graph(X,y,directed=directed)
    triples, triples0 = graph.get_triples(p, n, True)
    return triples, triples0
