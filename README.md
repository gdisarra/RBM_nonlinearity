# RBM_nonlinearity
Code for training and analyzing the role of activation functions in Restricted Boltzmann Machine learning.

RBM training can be performed with four different activation functions of the hidden layer, in various ways:

## Lattice gas generative model
RBM with different activation functions is trained from a lattice gas model $p(\vec{ v })= \frac{1}{Z}\exp \left[-\sum_{i<j} J_{ij} v_i  v_j\right]$, by maximizing the exact log-likelihood (cross-entropy).

## RBM generative model
RBM with different activation functions is trained from a ground truth RBM $p(\vec{v})= \frac{1}{Z} \sum_{z} \exp \left[ -\sum_{i} b_i^* v_i - \sum_{i<\mu} w_{i\mu}^* v_i  z_{\mu} -\sum_{\mu} c_{\mu}^* z_{\mu}   \right]$, by maximizing the exact log-likelihood (cross-entropy).

### Standard training
A new RBM is trained with different activation functions and standard initial conditions.
### Ground truth training
An RBM with ground truth initial conditions - and perturbations of it - is trained with different activation functions, to explore the neighborhood of the optimal solution.

## MNIST
Contrastive Divergence (CD), Persistent CD or Alternating Gibbs sampling are developed to learn the MNIST dataset.

## Code description

### rbm_utils.py
Objects and method definition
#### Lattice Gas class
The lattice gas model can be defined, the Q parameters and partition function can be computed.
#### RBM class
The RBM can be defined, initialized and perturbed, Q parameters and the partition function can be computed, and the training algorithm is defined.
#### ground truth training
Trains the RBM from the ground truth probability distribution.


### rbm_analysis.py
Performs exact training from binary datasets generated from a lattice gas model or an RBM, calling functions in rbm_utils.py.

### sampling_functions.py
Contains helper functions for setting up different activation functions.
