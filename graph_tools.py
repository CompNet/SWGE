#! /usr/bin/env python

import os, glob, shutil, string, pickle, gzip
import igraph
from distrib import *
# import matplotlib.pyplot as plt
from igraph.drawing import *

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

import json
from misc import *

import networkx as nx
from xml.etree import ElementTree as et
et.register_namespace('', "http://www.gexf.net/1.1draft")

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

def convert_to_gefx(filename, name):
    g = nx.read_graphml(filename)
    basename = ".".join(filename.split(".")[:-1])
    sliceno = int(name.split("/")[1])
    g.graph['mode'] = 'slice'
    g.graph['timerepresentation'] = "timestamp"
    g.graph['timestamp'] = sliceno
    print (g.graph)
    nx.write_gexf(g, basename + ".gexf")



def incr_gexf(filename, name):
    basename = ".".join(filename.split(".")[:-1])
    filename = basename + ".gexf"
    sliceno = name.split("/")[1]
    tree = et.parse(filename)
    root = tree.getroot()
    graph = root[0]
    graph.attrib['mode'] = "slice"
    graph.attrib['timerepresentation'] = "timestamp"
    graph.attrib['timestamp'] = sliceno
    edges = graph.getchildren()[-1]
    for i in range(len(edges)):
        e = edges[i]
        del e.attrib['id']

    tree.write(filename, encoding='UTF-8', xml_declaration=True)
    



def place_vertices(g, layout):
    i = 0
    for v in g.vs:
        v['x'] = layout[i][0]
        v['y'] = layout[i][1]

def polarity_comment(comment):
    try:
        if nlp(str(comment))[0]["label"] == "NEGATIVE":
            return -1
        return 1
    except:
        return 1


#build classic graph + signed graph + positive and negative graphs
def build_graphs(rows, window_size=10, distrib='spread', directed=False):

    g_classic = igraph.Graph(directed=directed)
    g_classic['vnames'] = set()
    g = igraph.Graph(directed=directed)
    g['vnames'] = set()
    g_positive = igraph.Graph(directed=directed)
    g_positive['vnames'] = set()
    g_negative = igraph.Graph(directed=directed)
    g_negative['vnames'] = set()
    for i in range(len(rows)):
        # Fixed-length window of messages before to update the edges
        w_start = i - window_size
        if w_start < 0:
            w_start = 0
        w = rows[w_start : i]
        target_date, target_uid, target_message = rows[i]
        comment_polarity = polarity_comment(target_message)
        # Check if message author in the graph, else add him.
        if target_uid not in g['vnames']:
            g.add_vertex(name=str(target_uid))
            g['vnames'].add(target_uid)
        target_vertex = g.vs.find(name=str(target_uid))
        if target_uid not in g_classic['vnames']:
            g_classic.add_vertex(name=str(target_uid))
            g_classic['vnames'].add(target_uid)
        target_vertex_classic = g_classic.vs.find(name=str(target_uid))

        if comment_polarity == 1:
            if target_uid not in g_positive['vnames']:
                g_positive.add_vertex(name=str(target_uid))
                g_positive['vnames'].add(target_uid)
            target_vertex_positive = g_positive.vs.find(name=str(target_uid))

        if comment_polarity == -1:
            if target_uid not in g_negative['vnames']:
                g_negative.add_vertex(name=str(target_uid))
                g_negative['vnames'].add(target_uid)
            target_vertex_negative = g_negative.vs.find(name=str(target_uid))
        weights = get_weigths(get_targets(w, target_uid), strategy = distrib)
        for to, weight in weights:
            #signed graph
            v_to = g.vs.find(name=str(to))
            eid = g.get_eid(target_vertex, v_to, directed=directed, error=False)
            if eid == -1: # edge does not exist
                g.add_edges([(target_vertex, v_to)])    # add it
                eid = g.get_eid(target_vertex, v_to, directed=directed, error=False)
                g.es[eid]["weight"] = comment_polarity*weight # and specify the weight
            else:
                g.es[eid]["weight"] += comment_polarity*weight
            #classic graph
            v_to_classic = g_classic.vs.find(name=str(to))
            eid = g_classic.get_eid(target_vertex_classic, v_to_classic, directed=directed, error=False)
            if eid == -1: # edge does not exist
                g_classic.add_edges([(target_vertex_classic, v_to_classic)])    # add it
                eid = g_classic.get_eid(target_vertex_classic, v_to_classic, directed=directed, error=False)
                g_classic.es[eid]["weight"] = weight # and specify the weight
            else:
                g_classic.es[eid]["weight"] += weight
            ###positive
            if comment_polarity == 1:
                if to not in g_positive['vnames']:
                    g_positive.add_vertex(name=str(to))
                    g_positive['vnames'].add(to)
                v_to_positive = g_positive.vs.find(name=str(to))
                eid = g_positive.get_eid(target_vertex_positive, v_to_positive, directed=directed, error=False)
                if eid == -1: # edge does not exist
                    g_positive.add_edges([(target_vertex_positive, v_to_positive)])    # add it
                    eid = g_positive.get_eid(target_vertex_positive, v_to_positive, directed=directed, error=False)
                    g_positive.es[eid]["weight"] = comment_polarity*weight # and specify the weight
                else:
                    g_positive.es[eid]["weight"] += comment_polarity*weight
            ###negative
            if comment_polarity == -1:
                if to not in g_negative['vnames']:
                    g_negative.add_vertex(name=str(to))
                    g_negative['vnames'].add(to)
                v_to_negative = g_negative.vs.find(name=str(to))
                eid = g_negative.get_eid(target_vertex_negative, v_to_negative, directed=directed, error=False)
                if eid == -1: # edge does not exist
                    g_negative.add_edges([(target_vertex_negative, v_to_negative)])    # add it
                    eid = g_negative.get_eid(target_vertex_negative, v_to_negative, directed=directed, error=False)
                    g_negative.es[eid]["weight"] = comment_polarity*weight # and specify the weight
                else:
                    g_negative.es[eid]["weight"] += comment_polarity*weight
    return g, g_classic, g_positive, g_negative




if __name__ == "__main__":
    import doctest
    doctest.testmod()




