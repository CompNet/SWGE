from SG2V import *
from WSGCN import *
from SGCN import *
from SiNE import *
from evaluation import *


run_all_experiments_sg2v()
run_all_experiments_wsgcn()
run_sgcn()
run_sine()


#evaluation
embeddings = load_embeddings(path_emb)
labels = load_labels(path_label)
evaluation(embeddings, labels)
