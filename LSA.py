from __future__          import division
from scipy.sparse.linalg import svds

import numpy as np

import pickle
import os



class LSA(object):
	def __init__(self, documents, tdm_path="TDM.pickle"):
		self.documents = documents
		
		if os.path.isfile(tdm_path):
			print("Loading term-document matrix ...")
			with open(tdm_path, 'r') as file:
				self.TDM = pickle.load(file)

		else:
			self.TDM = self.getTDMatrix()

			with open(tdm_path, 'wb') as file:
				pickle.dump(self.TDM, file, protocol=pickle.HIGHEST_PROTOCOL)

	'''
		This method is used to setup all the variables related to terms/words 
		that will be usefull for the rest of the computation.

		To reduce the number of terms, we only concidere words that appears
		in more than one document
	'''
	def setupWords(self):
		words = list(set([word for doc in self.documents for word in doc]))
		words_reverse_dictionary = {word:i for i, word in enumerate(words)}

		word_count_documents = np.zeros(len(words))
		for document in self.documents:
			for word in set(document):
				word_count_documents[words_reverse_dictionary[word]] += 1

		words = [word for i, word in enumerate(words) if word_count_documents[i] > 1 ]
		words_reverse_dictionary = {word:i for i, word in enumerate(words)}
		word_count_documents = np.array([count for count in word_count_documents if count > 1])

		return words, words_reverse_dictionary, word_count_documents


	def getTDMatrix(self):
		print("Getting Idf ...")
		words, words_reverse_dictionary, word_count_documents = self.setupWords()
		idf = np.log(len(self.documents)/word_count_documents)

		print("Creating the sparse term-document matrix ...")
		term_document_matrix = np.zeros((len(words),len(self.documents)))

		for j, document in enumerate(self.documents):
			for word in document:
				if word in words:
					i = words_reverse_dictionary[word]

					term_document_matrix[i,j] += idf[i]

		return term_document_matrix

	'''
	def cosine(self, a, b):
		#to avoid problems with division by zero
		if (np.linalg.norm(a) * np.linalg.norm(b)) == 0:
			return 0

		return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))
	
	# returns the distances between the document at the given index and all the documents 
	def getDistances(self, index, rank):
		print("Performing SVD ...")
		Uk, Sk,Vkt = svds(self.TDM, rank)

		print("Calculating distances between documents ...")
		vectors = np.dot(np.diag(Sk),Vkt)
		distances = []
		for j in range(vectors.shape[1]):
			distances.append(self.cosine(vectors[:,index], vectors[:,j]))

		return np.array(distances)
	'''

	# returns the distances between the document at the given index and all the documents
	def getDistances(self, index, rank):
		print("Performing SVD ...")
		Uk, Sk,Vkt = svds(self.TDM, rank)

		print("Calculating distances between documents ...")
		vectors = np.dot(np.diag(Sk),Vkt)
		c = vectors[:,index]

		# to avoid division by zero
		normC = np.linalg.norm(c) if np.linalg.norm(c) > 0 else 1
		normVectors = np.linalg.norm(vectors, axis=0)
		normVectors[np.where(normVectors == 0)] = 1
		
		#normalization
		c = c / normC
		vectors = vectors / normVectors

		return np.dot(c,vectors)


