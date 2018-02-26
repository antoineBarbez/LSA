import numpy as np
from scipy.linalg import svd


class LSA(object):
	def __init__(self, documents):
		self.documents = documents
		self.words = list(set([word for doc in documents for word in doc]))
		self.words_reverse_dictionary = {word:i for i, word in enumerate(self.words)}

		self.idf = self.getIdf(self.words, self.documents)
		self.TDM = self.getTDMatrix()


	def getIdf(self, words, documents):
		inverse_proportion = np.ones(len(self.words)) * len(self.documents)

		for i, word in enumerate(words):
			count = 0
			for document in self.documents:
				if word in document:
					count += 1

			inverse_proportion[i] = inverse_proportion[i]/count

		return np.log(inverse_proportion)

	def getTDMatrix(self):
		term_document_matrix = np.zeros((len(self.words),len(self.documents)))

		for j, document in enumerate(self.documents):
			for word in document:
				i = self.words_reverse_dictionary[word]

				term_document_matrix[i,j] += self.idf[i]

		return term_document_matrix

	def getTDMApprox(self, rank):
		U, S, Vt = svd(self.TDM, full_matrices=False)
		
		minIndexes = np.argsort(S)[:-rank]

		Uk = np.delete(U, minIndexes, 1)
		Sk = np.delete(S, minIndexes)
		Vkt = np.delete(Vt, minIndexes, 0)





if __name__ == "__main__":
	lsa = LSA([['a','b','a'],['a','b','c','b'],['a','d']])

	#print(lsa.getTDMatrix())

	lsa.getTDMApprox(2)

	#a = np.array([1,2,3,4,5])

	#print(np.argsort(a)[-2:])

