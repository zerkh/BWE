import theano.tensor as T
import numpy as np

class GCWE:
	def __init__(self, window_size=5, word_dim=25, hidden_dim=100,\
			global_hidden_size=100):
		self.window_size = window_size
		self.hiddem_dim = hidden_dim
		self.word_dim = word_dim

		W1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),\
					(word_dim*window_size, hidden_dim))
		b1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),\
					(1, hidden_dim))
		W2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),\
					(hidden_dim, 1))
		b2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),\
					(1, 1))

		Wg1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),\
					(word_dim*2, hidden_dim))
		bg1 = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim),\
					(1, hidden_dim))
		Wg2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),\
					(hidden_dim, 1))
		bg2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),\
					(1, 1))

		__build_GCWE__()

	def __build_GCWE__():
		
