from __future__ import division

import numpy as np
from scipy.sparse.linalg import svds



class LSA(object):
	def __init__(self, documents):
		self.documents = documents
		self.setupWords()

		self.idf = np.log(len(self.documents)/self.word_count_documents)
		self.TDM = self.getTDMatrix()


	def setupWords(self):
		words = list(set([word for doc in self.documents for word in doc]))
		words_reverse_dictionary = {word:i for i, word in enumerate(words)}

		word_count_documents = np.zeros(len(words))
		for document in self.documents:
			for word in set(document):
				word_count_documents[words_reverse_dictionary[word]] += 1

		self.words = [word for i, word in enumerate(words) if word_count_documents[i] > 1 ]
		self.words_reverse_dictionary = {word:i for i, word in enumerate(self.words)}
		self.word_count_documents = np.array([count for count in word_count_documents if count > 1])


	def getTDMatrix(self):
		print("Creating the sparse term-document matrix ...")
		term_document_matrix = np.zeros((len(self.words),len(self.documents)))

		for j, document in enumerate(self.documents):
			for word in document:
				if word in self.words:
					i = self.words_reverse_dictionary[word]

					term_document_matrix[i,j] += self.idf[i]

		return term_document_matrix

	def cosine(self, a, b):
		if (np.linalg.norm(a) * np.linalg.norm(b)) == 0:
			return 0

		return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))

	def getDistances(self, index, rank=100):
		print("Performing SVD ...")
		Uk, Sk,Vkt = svds(self.TDM, rank)

		print("Calculating distances between documents ...")
		vectors = np.dot(np.diag(Sk),Vkt)
		distances = []
		for j in range(vectors.shape[1]):
			distances.append(self.cosine(vectors[:,index], vectors[:,j]))

		return np.array(distances)


if __name__ == "__main__":
	lsa = LSA([['a','b','a'],['a','b','c','b'],['a','d']])

	#print(lsa.getDistances(0,3))

	#lsa.getTDMApprox(2)

	#a = np.array([1,2,3])
	#b = np.array([[1,2,3],[4,5,6],[7,8,9]])
	#print(range(3))


	#a = 5/np.array([1,2,3,4,5])

	#print(a)
