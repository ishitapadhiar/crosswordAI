import numpy as np
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
	train_clues = np.load('squashed_clues_train.bin.npy')
	train_answers = np.load('answers_train.bin.npy')
	test_clues = np.load('squashed_clues_test.bin.npy')
	test_answers = np.load('answers_test.bin.npy')
	print(train_clues)
	print(train_answers)
	# Build model (sequential layered NN)
	model = keras.Sequential([
		keras.layers.Dense(10, input_shape=(5,)),
		keras.layers.Dense(10, activation=tf.nn.relu),
		keras.layers.Dense(10, activation=tf.nn.relu),
		keras.layers.Dense(5, activation=tf.nn.softmax)
	])
	model.compile(
		optimizer='adam',
		loss=tf.losses.softmax_cross_entropy,
		metrics=['accuracy']
	)
	model.fit(train_clues, train_answers, epochs=30)
	ret = model.evaluate(test_clues, test_answers)
	print(ret)
