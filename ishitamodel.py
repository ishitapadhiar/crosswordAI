import tensorflow as tf
from glove import Corpus, Glove
import glove.corpus
import sys

def txtToLst(file):
	f = open(file,'r')
	midLst = f.readlines()
	i = 0
	while i < len(midLst):
		midLst[i] = midLst[i].strip()
		if len(midLst[i]) == 0:
			midLst.pop(i)
		i +=1

	f.close()

	finLst = list()
	for i in range(len(midLst)):
		finLst.append(midLst[i].split())

	return finLst


if __name__ == '__main__':
	clueLst = txtToLst(sys.argv[1])
	print(len(clueLst))
	ansLst = txtToLst(sys.argv[2])
	print(len(ansLst))

	corpus = Corpus()
	corpus.fit(clueLst, window = 10)
	glove = Glove(no_components = 5, learning_rate = 0.05)
	glove.fit(corpus.matrix, epochs = 30, no_threads = 4, verbose = True)
	glove.add_dictionary(corpus.dictionary)
	glove.save('clue.model')

	print (glove.word_vectors[glove.dictionary['happy']])


	corpus1 = Corpus()
	corpus1.fit(ansLst, window = 10)
	glove1 = Glove(no_components = 5, learning_rate = 0.05)
	glove1.fit(corpus1.matrix, epochs = 30, no_threads = 4, verbose = True)
	glove1.add_dictionary(corpus1.dictionary)
	glove1.save('answer.model')

	print (glove1.word_vectors[glove1.dictionary['ERA']])
