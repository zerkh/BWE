import theano.tensor as T
import numpy as np

class GCWE:
	def __init__(self, configer = None):
		self.window_size = configer.window_size
		self.hiddem_dim = configer.hidden_dim
		self.word_dim = configer.word_dim
		self.global_hidden_dim = configer.global_hidden_dim

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

		self.W1 = theano.shared(name="W1", value=W1.astype(theano.config.floatX))
		self.b1 = theano.shared(name="b1", value=b1.astype(theano.config.floatX))
		self.W2 = theano.shared(name="W2", value=W2.astype(theano.config.floatX))
		self.b2 = theano.shared(name="b2", value=b2.astype(theano.config.floatX))
		
		self.Wg1 = theano.shared(name="Wg1", value=Wg1.astype(theano.config.floatX))
		self.bg1 = theano.shared(name="bg1", value=bg1.astype(theano.config.floatX))
		self.Wg2 = theano.shared(name="Wg2", value=Wg2.astype(theano.config.floatX))
		self.bg2 = theano.shared(name="bg2", value=bg2.astype(theano.config.floatX))

		__build_GCWE__()

	def __build_GCWE__():
		pass
