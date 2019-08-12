import tensorflow as tf
from glove import Corpus, Glove
import glove.corpus

def tsvToLst(file):
	f = open(file,'r')
	midLst = f.readlines()
	f.close()


	clu = list()
	ans = list()
	length = list()
	for i in range(len(midLst)):
		tempLst = midLst[i].split('\t')
		clu.append(tempLst[0])
		ans.append(tempLst[1])
		length.append(tempLst[2])

	return clu, ans, length 





clues, answers, length = tsvToLst(sys.argv[1])
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

print (glove1.word_vectors[glove1.dictionary['era']])
