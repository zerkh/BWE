import theano.tensor as T
import theano
import numpy as np
from config import GCWEConfiger

class GCWE:
	def __init__(self, configer = None):
		self.window_size = configer.window_size
		self.hidden_dim = configer.hidden_dim
		self.word_dim = configer.word_dim
		self.global_hidden_dim = configer.global_hidden_dim

		W1 = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim),\
					(self.word_dim*self.window_size, self.hidden_dim))
		b1 = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim),\
					(1, self.hidden_dim))
		W2 = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim),\
					(self.hidden_dim, 1))
		b2 = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim),\
					(1, 1))

		Wg1 = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim),\
					(self.word_dim*2, self.hidden_dim))
		bg1 = np.random.uniform(-np.sqrt(1./self.word_dim), np.sqrt(1./self.word_dim),\
					(1, self.hidden_dim))
		Wg2 = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim),\
					(self.hidden_dim, 1))
		bg2 = np.random.uniform(-np.sqrt(1./self.hidden_dim), np.sqrt(1./self.hidden_dim),\
					(1, 1))

		self.W1 = theano.shared(name="W1", value=W1.astype(theano.config.floatX))
		self.b1 = theano.shared(name="b1", value=b1.astype(theano.config.floatX))
		self.W2 = theano.shared(name="W2", value=W2.astype(theano.config.floatX))
		self.b2 = theano.shared(name="b2", value=b2.astype(theano.config.floatX))
		
		self.Wg1 = theano.shared(name="Wg1", value=Wg1.astype(theano.config.floatX))
		self.bg1 = theano.shared(name="bg1", value=bg1.astype(theano.config.floatX))
		self.Wg2 = theano.shared(name="Wg2", value=Wg2.astype(theano.config.floatX))
		self.bg2 = theano.shared(name="bg2", value=bg2.astype(theano.config.floatX))

	def forward(self, word_emb, X_local, X, X_g):
		input_layer = T.concatenate([word_emb[X_local[0]],word_emb[X_local[1]],\
					word_emb[X_local[2]],word_emb[X_local[3]],\
					word_emb[X]])

		hidden_layer = T.tanh(input_layer.dot(self.W1)+self.b1)
		score_local = hidden_layer.dot(self.W2)+self.b2

		global_input_layer = T.concatenate([word_emb[X], X_g], axis=0)
		global_hidden_layer = T.tanh(global_input_layer.dot(self.Wg1)+self.bg1)
		score_global = global_hidden_layer.dot(self.Wg2)+self.bg2
		
		score = score_local + score_global

		return score[0][0]

	def target_function(self, x_neg, x_local, x, x_g):
		score = self.forward(word_emb, x_local, x, x_g)
		score_neg = self.forward(word_emb, x_local, x_neg, x_g)
			
		return T.max([0, 1-score+score_neg])

	def train(self, word_emb):
		X_local = T.ivector(name="X_local")
		X = T.iscalar(name="X")
		X_neg = T.ivector(name="X_neg")
		X_g = T.dvector(name="X_g")
		
		[o_error], updates = theano.scan(self.target_function, sequences=X_neg,\
										non_sequences=[X_local, X, X_g])
		
		error_sum = T.sum(o_error)
		self.c_error = theano.function([X_local, X, X_neg, X_g], error_sum)
		
		d_word_emb = T.grad(error_sum, word_emb)
		d_W1 = T.grad(error_sum, self.W1)
		d_b1 = T.grad(error_sum, self.b1)
		d_W2 = T.grad(error_sum, self.W2)
		d_b2 = T.grad(error_sum, self.b2)
		d_Wg1 = T.grad(error_sum, self.Wg1)
		d_bg1 = T.grad(error_sum, self.bg1)
		d_Wg2 = T.grad(error_sum, self.Wg2)
		d_bg2 = T.grad(error_sum, self.bg2)
		
		self.train_step = theano.function([X_local, X, X_neg, X_g], [], \
										updates=[(self.W1-d_W1),
												(self.b1-d_b1),
												(self.W2-d_W2),
												(self.b2-d_b2),
												(self.Wg1-d_Wg1),
												(self.bg1-d_bg1),
												(self.Wg2-d_Wg2),
												(self.bg2-d_bg2)])
		
if __name__ == "__main__":
	configer = GCWEConfiger("config.conf")
	model = GCWE(configer)
	
	print model.W1.get_value()
	
	X_local = np.array([0,1,2,3,4])
	X=5
	X_neg = [0,1,2,3,4,5]
	
	emb = np.random.uniform(-np.sqrt(1./25), np.sqrt(1./25), (10, 25))
	word_emb = theano.shared(name="word_emb", value=emb.astype(theano.config.floatX))
	X_g = np.mean(emb, axis=0)
	
	model.train(word_emb)
	model.train_step(X_local, X, X_neg, X_g)
	
	print model.W1.get_value()