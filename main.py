from SG2V import *
from WSGCN import *
from evaluation import *


run_all_experiments_sg2v()
run_all_experiments_wsgcn()

#evaluation
embeddings = load_embeddings(path_emb)
labels = load_labels(path_label)
evaluation(embeddings, labels)
