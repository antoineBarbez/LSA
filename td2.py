
import numpy as np

import re
import sys
import argparse
import os
import nltk
import time
import LSA

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("input")
	parser.add_argument("--nb", type=int, default=5)
	parser.add_argument("--dim", type=int, default=70)
	return parser.parse_args()


#return an array whose elements are the words contained in the input string
def getWords(inputString):
	WORD_SEPARATOR = r'(?:(?:&nbsp)?[\s.,:;?!()\\/\'\"])+'
	words = re.split(WORD_SEPARATOR, inputString.strip().lower())

	#stemming + remove the words of 1 or 2 letters
	stemmer = nltk.stem.snowball.FrenchStemmer()
	words = [stemmer.stem(x) for x in words if len(x) > 2]

	return words

'''
	Returns the nbRec closest courses from the one given in argument
'''
def getClosestClasses(classCode, nbRec, vectorsDim):
	print("Preprocessing ...")

	codes = []
	for path,dirs,files in os.walk('PolyHEC/'):
		for f in files:
			classId = f.split('.')[0]
			codes.append(classId)

	if nbRec > len(codes):
		print("You ask for more recommendations than the total number of courses")
		print("--nb will be set to: " + str(len(codes)))
		nbRec = len(codes)

	codes_reverse_dictionary = {code:i for i, code in enumerate(codes)}

	documents = [getWords(getDescription(code)) for code in codes]
	lsa = LSA.LSA(documents)

	# to free memory
	del documents

	if vectorsDim > len(lsa.words):
		print("--dim is too large, will be set to: " + str(len(lsa.words)))

	distances = lsa.getDistances(codes_reverse_dictionary[classCode.upper()], vectorsDim)
	maxIndexes = np.argsort(distances)[-nbRec:][::-1]

	CC = [{"classCode" : codes[idx], "distance": distances[idx]} for idx in maxIndexes]

	return CC

# Returns the title of a course
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

# Returns the description of a course
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
	nbRec = args.nb
	vectorsDim = args.dim

	outputFilePath = classCode.upper() + ".txt"

	outputFile = open(outputFilePath,"w")
	outputFile.write("Compared class : " + classCode.lower() + "\n")
	outputFile.write("Titre : " + getTitle(classCode) + "\n")
	outputFile.write("Description: " + getDescription(classCode) + "\n")
	outputFile.write("\n")

	for cour in getClosestClasses(classCode, nbRec, vectorsDim):
		outputFile.write(cour["classCode"].lower() + " : " + str(cour["distance"]) + "\n")
		outputFile.write("Titre : " + getTitle(cour["classCode"]) + "\n")
		outputFile.write("Description : " + getDescription(cour["classCode"]) + "\n")
		outputFile.write("\n")
			

	outputFile.close()



if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("This script has to be used as :\n\tpython td2.py classCode")
		sys.exit(1)
		
	main()