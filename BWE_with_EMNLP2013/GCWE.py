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

	def forward(self):

		W1,b1,W2,b2,Wg1,bg1,Wg2,bg2 = \
		self.W1, self.b1, self.W2, self.b2,\
		self.Wg1, self.bg1, self.Wg2, self.bg2

		input_layer = T.catenate(X[0].dot(word_emb),X[1].dot(word_emb),\
					X[2].dot(word_emb),X[3].dot(word_emb),\
					X[4].dot(word_emb),axis=0)

		hidden_layer = T.tanh(input_layer.dot(W1)+b1)
		score_local = hidden_layer.dot(W2)+b2

		global_input_layer = T.catenate(X[4], X_g, axis=0)
		global_hidden_layer = T.tanh(global_input_layer.dot(Wg1)+bg1)
		score_global = global_hidden_layer.dot(Wg2)+b2g
		
		score = score_local + score_global

		self.prediction = theano.function([X,word_emb,X_g], score)
		
