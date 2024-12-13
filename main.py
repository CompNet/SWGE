from SG2V.SG2V import run_all_experiments_sg2v
from WSGCN.run_wsgcn import run_all_experiments_wsgcn
#from SGCN.src.main import *
from evaluation import *


run_all_experiments_sg2v()
#run_all_experiments_wsgcn()
#run_sgcn()


#evaluation

evaluate_SG2V()
