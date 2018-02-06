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
  
  for i in range(0, X.shape[0]):
    score_sum = 0
    correct_score = 0
    for j in range(0, W.shape[1]):
        score = np.exp(image * W[:,j])
        score_sum += score
        if j == y[i]:
            correct_score = score
    p = correct_score / score_sum
    loss -= np.ln(p)
    for j in range(0, W.shape[1]):
        dW[:, j] += p * X[i]
        if j == y[i]:
            dW[:, j] -= X[i]
  loss /= X.shape[0]
    
  # Regularization
  loss += reg * np.sum(W * W)
  dW += .5(reg * W)
    
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.exp(X.dot(W))
  P = scores / scores.sum(axis = 1)
  loss = np.sum(np.log(p)) / X.shape[0]
  dW = (X.T).dot(P)
  dW[:, y] -= X
  
  # Regularization
  loss += reg * np.sum(W * W)
  dW += .5(reg * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

