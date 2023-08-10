from SiNEmaster.graph import *
from SiNEmaster.stemmodels import SiNE, fit_sine_model as fit_model
import pickle

#pickled list of labels
labels_path = "labels.pickle"
graphs_path = "data/CCS"

embeddings = []
labels = []
with open(labels_path, "rb") as f:
	lb = pickle.load(f)


for i in range(2545):
	try:
		graph = Graph.read_from_file("%s/%s.csv" %(graphs_path, i), delimiter=',', directed=True)
		if len(graph.get_positive_edges()) + len(graph.get_negative_edges()) > 1:

			model = fit_model(
						num_nodes=len(graph),
						dims_arr=[32, 32],
						triples=graph.get_triplets(),
						triples0=None,
						delta=1.0,
						delta0=0.5,
						batch_size=300,
						batch_size0=300,
						epochs=30,
						lr=0.01,
						lam=0.0001,
						lr_decay=0.0,
						p=2,
						print_loss=False,
						p0=False,
					)

			embedding = model.get_x()
			embedding = embedding.detach().numpy().tolist()[0]
			embeddings.append(embedding)
			labels.append(lb[i])
			print (i)
	except:
		print ("error")

with open("out/SiNE/sine_embeddings.pkl", "wb") as f:
	pickle.dump(embeddings, f)