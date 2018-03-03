import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mdA import mdA
from utils import load_data, save_data


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))

        if y.dtype.startswith('int'):

            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=theano.tensor.nnet.sigmoid):
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)

            b = theano.shared(b_values, name='b', borrow=True)


        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

# start-snippet-1
class mSdA(object):
    """Marginalized Stacked denoising auto-encoder class (mSdA)

    A marginalizedstacked denoising autoencoder model is obtained by stacking several
    marginalized mdAs. The hidden layer of the mdA at layer `i` becomes the input of
    the mdA at layer `i+1`. The first layer mdA gets as input the input of
    the mSdA, and the hidden layer of the last mdA represents the output.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):

        self.sigmoid_layers = []
        self.mdA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels


        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the mSdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)

            # Construct a mDA that shared weights with this layer
            mdA_layer = mdA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.mdA_layers.append(mdA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for fine-tuning,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the mdA corresponding to the layer with same index.
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for mdA in self.mdA_layers:
            # get the cost and the updates list
            cost, updates = mdA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index, theano.In(corruption_level, value=0.2),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

def test_mSdA(finetune_lr=0.1, pretraining_epochs=200,
             pretrain_lr=0.01, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=16):
    """
    Demonstrates how to train and test a marginalized denoising autoencoder.

    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')

    msda = mSdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000],
        n_outs=10
    )

    # PRETRAINING THE MODEL

    print('... getting the pretraining functions')
    pretraining_fns = msda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = time.clock()

    ## Pre-train layer-wise
    corruption_levels = [.5, .5]
    for i in range(msda.n_layers):

        # go through pretraining epochs
        for epoch in range(pretraining_epochs):

            # go through the training set
            c = []

            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, 
                    corruption=corruption_levels[i], lr=pretrain_lr))
                end_time = time.clock()


            if epoch%15 == 0:
                save_data(i+1, pretrain_lr, corruption_levels[i], epoch+1,
                    msda.mdA_layers[i].W.get_value(borrow=True), 
                    msda.mdA_layers[i].b.get_value(borrow=True), end_time - start_time);

            print('Pre-training layer %i, epoch %d, cost %f' % (i+1, epoch+1, numpy.mean(c, dtype='float64')))

    end_time = time.clock()

    print('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # FINETUNING THE MODEL

    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = msda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches 
    patience_increase = 2.  
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    for i in range(msda.n_layers):
                        save_data(i+1, finetune_lr, corruption_levels[i], epoch+1, 
                            msda.sigmoid_layers[i].W.get_value(borrow=True), 
                            msda.sigmoid_layers[i].b.get_value(borrow=True),
                            time.clock() - start_time, training_type="fine-tuning");

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_mSdA(dataset='/data/basic.pkl.gz')