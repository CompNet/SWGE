import subprocess
import glob


def run_all_SGCN(graphs_path):
	for i in range(len(glob.glob("%s/*.csv" % (graphs_path)))):
		command = "python SGCN-master/src/main.py --layers 32 --learning-rate 0.01 --reduction-dimensions 64 --epochs 10 --reduction-iterations 10 --edge-path %s/%s.csv --embedding-path output/SGCN/%s.csv --regression-weights-path /output/SGCN/weights/%s.csv" % (graphs_path, i, i ,i)
		process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
		output, error = process.communicate()