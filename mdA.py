import os
import sys
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class mdA(object):

	def __init__(self, numpy_rng, theano_rng=None, input=None,
				 n_visible=784, n_hidden=1000,
				 W=None, bhid=None, bvis=None):

		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if not theano_rng:
			theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

		if not W:
			initial_W = numpy.asarray(numpy_rng.uniform(
					  low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
					  size=(n_visible, n_hidden)), dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W', borrow=True)

		if not bvis:
			bvis = theano.shared(value=numpy.zeros(n_visible,
										 dtype=theano.config.floatX),
								 borrow=True)

		if not bhid:
			bhid = theano.shared(value=numpy.zeros(n_hidden,
												   dtype=theano.config.floatX),
								 name='b',
								 borrow=True)

		self.W = W
		# b corresponds to the bias of the hidden layer
		self.b = bhid
		# b_prime corresponds to the bias of the visible layer
		self.b_prime = bvis
		# tied weights, therefore W_prime is W transpose
		self.W_prime = self.W.T
		self.theano_rng = theano_rng
		# if no input is given, generate a variable representing the input
		if input == None:
			# we use a matrix because we expect a minibatch of several
			# examples, each example being a row
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

		self.params = [self.W, self.b, self.b_prime]

	def get_hidden_values(self, input):
		""" Computes the values of the hidden layer """
		return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

	def get_reconstructed_input(self, hidden):
		"""Computes the reconstructed input given the values of the
		hidden layer

		"""
		return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

	def get_cost_updates(self, noiserate, learning_rate):
		""" This function computes the cost and the updates for one trainng
		step of the mdA """

		y = self.get_hidden_values(self.x)

		# Clipping for avoiding numerical instability
		z = T.clip(self.get_reconstructed_input(y), 0.00247262315663, 0.997527376843)

		# Cross-entropy Loss
		L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

		# Regularization term because of Implicit Denoising via Marginalization
		dy = y * (1 - y)
		dz = z * (1 - z)

		df_x_2 = T.dot(T.dot(dz, self.W * self.W) * dy * dy,  self.W_prime * self.W_prime)

		L2 = noiserate * noiserate * T.mean(T.sum(df_x_2, axis=1))

		# Final Objective Function
		cost = T.mean(L) + 0.5 * L2

		gparams = T.grad(cost, self.params)

		# generate the list of updates
		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append((param, param - learning_rate * gparam))

		return (cost, updates)