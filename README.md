# Marginalised SDA for Nonlinear Respresentations
Theano implementation of the paper [Marginalized Denoising Auto-encoders for Nonlinear Representations][main-paper] (ICML 2014). Also extended it to stack multiple layers (*mSDA*) of *mDA*s for classification tasks.

Other denoising techniques have longer training time and high computational demands. *mSDA* addresses the problem by implicitly denoising the raw input via Marginalization and, thus, is effectively trained on *infinitely* many training samples without explicitly corrupting the data. There are similar approaches but they have non-linearity or latent representations stripped away. This addresses the disadvantages of those approaches, and hence is a generalization of those works.

The code is inspired from [LISA Lab](lisa-lab) deep learning tutorials.

### Training
There are two steps for training the *Marginalized Stacked Denoising Auto-encoder* (*mSDA*), where each step is performed seperately:
  - **Greedy unsupervised pre-training**: The layerwise approach in which each layer is trained as a *mDA* by minimizing the error in reconstructing its input. The input of each layer is the output of the previous layer with the input of the first layer as the given raw input.
  - **Supervised fine-tuning**: The fine-tuning approach in which a Logistic Regression layer is added on top of the network (only encoders of the auto-encoders). And, the whole network is trained in a supervised manner, i.e., as a Multi-layer Perceptron (MLP) with the given target class.

Our stacked model essentially has two parts: **Multiple *mDA* layers** and an **MLP**. Each *mDA* layer share the weight matrix and the bias of its encoding part with its corresponding sigmoid layer in the MLP.

### Requirements
 - Python 2.7
 - Theano >= 0.9
 - NumPy
 - SciPy (for saving model checkpoints)

### Run
To train the demo model :
```sh
python mSdA.py 
```
**Note**: My demo used the *basic* dataset, which is a sub-sampled version of the *MNIST* dataset. And the hyperparameters can be seen in the code.
### Demo Results

The the error rates for the best results on the *basic* dataset are:
- **Valdition Set**: 2.70 %
- **Test Error**: 3.33%

Resulted filters of first layer during training:  
![Image Filter Gif](https://raw.githubusercontent.com/jaiagarwal/mSdA/master/image-filters.gif)  
The filters are continuously improving and learning specialized feature extractors.

### References
 - [Marginalized Denoising Auto-encoders for Nonlinear Representations][main-paper] (ICML 2014)
 - [Marginalizing Stacked Linear Denoising Autoencoders][stacked-paper] (JMLR 2015)
 - [LISA Lab][lisa-lab] deep learning tutorials
 - UFLDL Stanford: [Stacked Auto-encoders][stanford-tut]


   [main-paper]: <http://www.cse.wustl.edu/~mchen/papers/deepmsda.pdf>
   [lisa-lab]: <https://github.com/lisa-lab/DeepLearningTutorials>
   [stanford-tut]: <http://ufldl.stanford.edu/wiki/index.php/Stacked_Autoencoders>
   [stacked-paper]: <http://www.jmlr.org/papers/volume16/chen15c/chen15c.pdf>
