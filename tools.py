import pickle, gzip

#Generator for split numbers for many-fold evaluation
def split_gen(size = 10):
	for split in range(size):
		yield split

#Load a dumped object
def zload(fileName):
    with gzip.open(fileName, "rb") as f:
        items = pickle.load(f, encoding="bytes")
        return items
        return pickle.load(f)

def zdump(obj, fileName):
    with gzip.open(fileName, "wb") as f:
        pickle.dump(obj, f, 2)