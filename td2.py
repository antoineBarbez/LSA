
import numpy as np

import re
import sys
import argparse
import os
import nltk
import time

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("input")
	parser.add_argument("--nb_recommendation", type=int, default=5)
	return parser.parse_args()

#return an array whose elements are the words contained in the input string
def getWords(inputString):
	WORD_SEPARATOR = r'(?:(?:&nbsp)?[\s.,:;?!()\\/\'\"])+'
	words = re.split(WORD_SEPARATOR, inputString.strip().lower())

	#remove occurences of empty strings in the array
	stemmer = nltk.stem.snowball.FrenchStemmer()
	words = [stemmer.stem(x) for x in words if x != '']

	return words

def getBagOfWords():
	bagOfWords = []

	for path,dirs,files in os.walk('PolyHEC/'):
		for f in files:
			classCode = f.split('.')[0]
			description = getDescription(classCode)
			bagOfWords += getWords(description)

	return list(set(bagOfWords))

def getClosestClasses(classCode, nbRec):
	bagOfWords = getBagOfWords()
	words_reverse_dictionary = {word:i for i, word in enumerate(bagOfWords)}

	codes = []

	for path,dirs,files in os.walk('PolyHEC/'):
		for f in files:
			classCode = f.split('.')[0]
			codes.append(classCode)

	codes_reverse_dictionary = {code:i for i, code in enumerate(codes)}

	start_time = time.clock()
	term_document_matrix = np.zeros((len(bagOfWords),len(codes)), dtype=np.int16)
	for code in codes:
		j = codes_reverse_dictionary[code]

		description = getDescription(code)
		words = getWords(description)

		for word in words:
			i = words_reverse_dictionary[word]
			term_document_matrix[i,j] = term_document_matrix[i,j] + 1

	total_time = time.clock() - start_time

	print(total_time)

	CC = [
		{"classCode" : "CRI3260", "distance": 0.333},
		{"classCode" : "DCM7163", "distance": 0.450},
		{"classCode" : "DEI1211", "distance": 0.890},
		{"classCode" : "LIT3420", "distance": 0.760}
	]

	return CC

def getTitle(classCode):
	classFilePath = "PolyHEC/" + classCode.upper() + ".txt"

	titlePattern = re.compile("TitreCours")

	title = ""

	with open(classFilePath, "r") as classFile:
		for line in classFile:
			if re.match(titlePattern, line) :
				title = line.split(": ")[1]
				title = re.sub("\n", "", title)


	return title

def getDescription(classCode):
	classFilePath = "PolyHEC/" + classCode.upper() + ".txt"

	descPattern = re.compile("DescriptionCours")

	description = ""

	with open(classFilePath, "r") as classFile:
		for line in classFile:
			if re.match(descPattern, line) :
				description = line.split(": ")[1]
				description = re.sub("\n", "", description)


	return description

#the main function that uses the previous methods to create the output file
def main():
	args = parse_args()
	classCode = args.input
	nbRec = args.nb_recommendation

	outputFilePath = classCode.upper() + ".txt"

	outputFile = open(outputFilePath,"w")
	outputFile.write("Compared class : " + classCode.lower() + "\n")
	outputFile.write("Titre : " + getTitle(classCode) + "\n")
	outputFile.write("Description: " + getDescription(classCode) + "\n")
	outputFile.write("\n")

	for cour in getClosestClasses(classCode, 5):
		outputFile.write(cour["classCode"].lower() + " : " + str(cour["distance"]) + "\n")
		outputFile.write("Titre : " + getTitle(cour["classCode"]) + "\n")
		outputFile.write("Description : " + getDescription(cour["classCode"]) + "\n")
		outputFile.write("\n")
			

	outputFile.close()



if __name__ == "__main__":
	#main()

	text = "Ce cours vise a habiliter l'etudiant a utiliser des modeles diagnostiques complexes, a apprendre les theories des organisations dont sont issus les modeles diagnostiques courants, a identifier les principaux enjeux lies au changement organisationnel et a elaborer les programmes de gestion du changement qui tiennent compte de la complexite organisationnelle et de sa realite politique. Les principaux modeles et theories seront appliques a des cas concrets et vecus, qui permettront d'en apprecier les principaux facteurs de contingence et d'application. Principales theories des organisations et de gestion du changement, modeles diagnostiques pertinents, modeles d'organisation dynamique."

	array = getWords(text)

	print(array)
