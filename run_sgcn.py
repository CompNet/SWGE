import subprocess
import glob
import os
import networkx as nx
import csv

"""
  Transforms a graphml graph to the format required by the SGCN method. The generated edgelists are saved in the "edgelist" folder.
   
  :param graphs_path: path to the folder containing all graphml files.
  :return None
"""
def transform_to_edgelist(graphs_path):
    os.makedirs("%s/edgelist" % (graphs_path)) 
    for i in range(len(glob.glob("%s/*.graphml" % (graphs_path)))):
        G = nx.read_graphml("%s/%s.graphml" % (graphs_path, i))
        corresp_nodes = {}
        nb_nodes = 0
        for n in G.nodes:
            corresp_nodes[n] = nb_nodes
            nb_nodes += 1

        edgelist = []
        for u,v,d in G.edges(data=True):
            u = corresp_nodes[u]
            v = corresp_nodes[v]
            if d["weight"] >= 0.0:
                d = 1
            else:
                d = -1
            edgelist.append([u,v,d])

        header = ["Node id 1", "Node id 2", "Sign"]
        #write file
        with open("%s/edgelist/%s.csv" % (graphs_path, i), 'w') as f:
            writer_pos = csv.writer(f)
            writer_pos.writerow(header)
            writer_pos.writerows(edgelist)

"""
  Learns the SGCN representations of all graphs by running the SGCN script. First, it transforms graphs to an edgelist to match the format required by SGCN script.
   
  :param graphs_path: path to the folder containing all graphml files.
  :return None
"""
def run_all_SGCN(graphs_path):
    if not os.path.exists("%s/edgelist" % (graphs_path)):
        transform_to_edgelist(graphs_path)
    for i in range(len(glob.glob("%s/edgelist/*.csv" % (graphs_path)))):
        command = "python SGCN-master/src/main.py --layers 32 --learning-rate 0.01 --reduction-dimensions 64 --epochs 10 --reduction-iterations 10 --edge-path %s/edgelist/%s.csv --embedding-path out/SGCN/%s.csv --regression-weights-path out/SGCN/weights/%s.csv" % (graphs_path, i, i ,i)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

def run_all_experiments_sgcn():
    for graphs_path in ["data/CSS/", "data/EPF/", "data/SSO/"]:
        run_all_SGCN(graphs_path)

if __name__ == '__main__':
    dataset = "SSO"
    graphs_path = "data/%s/" % (dataset)

    run_all_SGCN(graphs_path)