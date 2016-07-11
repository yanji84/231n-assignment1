import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples, num_classes = X.shape
  for i in range(num_samples):
    # run forward pass on x[i]
    scores = X[i].dot(W)
    # for numerical stability
    scores -= np.max(scores)

    score_exp = np.exp(scores)
    normalization_constant = np.sum(score_exp)
    all_probs = score_exp / normalization_constant
    loss += -np.log(all_probs[y[i]])
    # gradient contributed by this example
    dx = all_probs 
    dx[y[i]] -= 1
    dW += np.dot(X[i].reshape(-1, 1), dx.reshape(1, -1))

  loss /= num_samples
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_samples
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_samples, num_classes  = X.shape

  # forward pass
  scores = X.dot(W)

  # for numerical stability
  scores -= np.max(scores)

  scores_exp = np.exp(scores)  
  normalization_constant = np.sum(scores_exp, axis=1, keepdims=True)

  all_probs = scores_exp / normalization_constant 

  # compute loss
  loss = -np.log(all_probs[range(num_samples), y]) 
  loss = np.sum(loss) / num_samples
  loss += 0.5 * reg * np.sum(W * W)

  # calculate gradient
  dx = all_probs
  dx[range(num_samples), y] -= 1
  dx /= num_samples
  dW = np.dot(X.T, dx)
  dW += reg * W

  return loss, dW

