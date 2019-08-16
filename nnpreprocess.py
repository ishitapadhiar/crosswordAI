import random
import numpy as np
from glove import Glove
from sklearn.decomposition import PCA
from ishitamodel import txtToLst

def getData(clues, answers, clue_txt, ans_txt):
	"""Processes the multivector clue data into a single vector using PCA.
		Returns the clue and answer data as numpy arrays."""
	train_clues = []
	train_answers = []
	test_clues = []
	test_answers = []
	pca = PCA(n_components=1)
	for i in range(len(clue_txt)):
		clue_vects = np.array([clues.word_vectors[clues.dictionary[w]] for w in clue_txt[i]])
		if len(clue_vects) == 0:
			# skip clues with no important words
			continue
		# transpose data for PCA function
		clue_vects = np.transpose(clue_vects)
		clue_vect = np.array(pca.fit_transform(clue_vects))
		# "untranspose" data
		clue_vect = clue_vect.reshape(len(clue_vect))
		# answers are simple, unlike clues
		ans_vect = answers.word_vectors[answers.dictionary[ans_txt[i][0]]]
		if random.random() < 0.9:
			train_clues.append(clue_vect)
			train_answers.append(ans_vect)
		else:
			test_clues.append(clue_vect)
			test_answers.append(ans_vect)
	return (np.array(train_clues), np.array(train_answers)), (np.array(test_clues), np.array(test_answers))

if __name__ == '__main__':
	clues = Glove.load('clue.model')
	answers = Glove.load('answer.model')
	clue_text = txtToLst('keywords.txt')
	ans_text = txtToLst('answers.txt')
	(train_clues, train_answers), (test_clues, test_answers) = getData(clues, answers, clue_text, ans_text)
	np.save('squashed_clues_train.bin', train_clues)
	np.save('squashed_clues_test.bin', test_clues)
	np.save('answers_train.bin', train_answers)
	np.save('answers_test.bin', test_answers)
	print(train_clues)
	print(train_answers)
