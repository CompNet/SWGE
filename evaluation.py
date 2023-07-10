import pickle as pkl
import glob
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def load_labels(path):
	with open(path, "rb") as f:
		labels = pickle.load(f)
	return labels

def load_embeddings(path):
	embeddings = []
	for i in range(len(glob.glob("%s/*.pkl" % (path)))):
		with open("%s/%s.pkl" % (path, i), "rb") as f:
			emb = pickle.load(f)
		embeddings.append(emb)
	return embeddings

def evaluation(embeddings, labels):
	X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.3, random_state=7)
	binary_classifier_model = svm.SVC(class_weight="balanced")
	binary_classifier_model.fit(X_train, y_train)

	y_pred = binary_classifier_model.predict(X_test)

	print ("Micro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='micro')))
	print ("Macro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='macro')))




embeddings = load_embeddings(path_emb)
labels = load_labels(path_label)
evaluation(embeddings, labels)