from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from karateclub import Graph2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

"""
    Adapted from https://github.com/benedekrozemberczki/SGCN
"""

class Signed_Graph2Vec(Graph2Vec):
    """
    Extends Graph2Vec class add a fit_documents method. 
    """

    def fit_documents(self, documents: List[gensim.models.doc2vec.TaggedDocument]): 
        """
        Fit Doc2Vec model directly with extracted features.
        """
        self.model = Doc2Vec(
                documents,
                vector_size=self.dimensions,
                window=0,
                min_count=self.min_count,
                dm=0,
                sample=self.down_sampling,
                workers=self.workers,
                epochs=self.epochs,
                alpha=self.learning_rate,
                seed=self.seed,
            )


class signed_WeisfeilerLehmanHashing(WeisfeilerLehmanHashing):
    """
    Extends Weisfeiler-Lehman feature extractor class to change the initialisation of nodes.
    Node labels are initialized with signed degrees.
    """

    def _set_features(self):
        """
        Creating the features.
        """
        if self.attributed:
            self.features = nx.get_node_attributes(self.graph, "feature")
        else:
            self.features = {}
            for node in self.graph.nodes():
                pos_deg = 0
                neg_deg = 0
                neighbors = self.graph.neighbors(node)
                for neb in neighbors:
                    if neb != node:
                        if self.graph.has_edge(node, neb):
                            edge_data = self.graph.get_edge_data(node, neb)
                            if edge_data["sign"] >= 0.0:
                                pos_deg += 1
                            elif edge_data["sign"] < 0.0:
                                neg_deg += 1
                f = str(pos_deg) + "|" + str(neg_deg)
                self.features[node] = f
        self.extracted_features = {k: [str(v)] for k, v in self.features.items()}



class WeisfeilerLehmanHashing_g2v(signed_WeisfeilerLehmanHashing):
    """
    "g2v" model. Basic G2V model.
    """

class WeisfeilerLehmanHashing_sg2vn(signed_WeisfeilerLehmanHashing):
    """
    "sg2vn" model.
    """

    def _do_a_recursion(self):
        """
        The method does a single WL recursion.

        Return types:
            * **new_features** *(dict of strings)* - The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.graph.nodes():
            nebs = self.graph.neighbors(node)
            degs = []
            for neb in nebs:
                if neb != node:
                    if self.graph.has_edge(node, neb):
                        edge_data = self.graph.get_edge_data(node, neb)
                        if edge_data["sign"] >= 0.0:
                            degs.append("+" + str(self.features[neb]))
                        elif edge_data["sign"] < 0.0:
                            degs.append("-" + str(self.features[neb]))
            ####


            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = {
            k: self.extracted_features[k] + [v] for k, v in new_features.items()
        }
        return new_features

class WeisfeilerLehmanHashing_sg2vsb(signed_WeisfeilerLehmanHashing):
    """
    "sg2vsb" model.
    """

    #TODO
    def _do_a_recursion(self):
        
