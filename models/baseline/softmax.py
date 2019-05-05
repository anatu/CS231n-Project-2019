from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Num classes = num cols of w
    C = W.shape[1]
    # Number of examples
    N = X.shape[0]
    
    dscores = np.zeros((N,C))
    
    for i in range(N):
        # Calculate the scores and subtract the max (predicted class)
        scores = np.dot(X[i], W)
        scores = scores - np.max(scores)
        # Calculate probability w/ numerical stability trick
        prob = np.exp(scores)/np.sum(np.exp(scores))
        
        binary_y = np.zeros(C)
        # Subtract the probabilities from the true example class for gradient
        binary_y[y[i]] = 1
        dscores[i] = prob - binary_y
        
        # Calculate loss for the example, add to total
        L_i = -np.log(prob[y[i]])
        loss = loss + L_i
    
    # Normalize and regularize loss
    loss = loss / N
    loss = loss + (0.5*reg*np.sum(W*W))
    
    # Calculate gradient, normalize / regularize 
    dW = np.dot(X.T, dscores)/N
    dW = dW + (reg*W)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = X.shape[0]
    C = W.shape[1]
    
    # Calculate scores and probability (increase dimension of max for broadcasting)
    scores = np.dot(X, W)
    scores = scores - np.max(scores, axis=1)[:, np.newaxis]
    prob = np.exp(scores)/np.sum(np.exp(scores), axis=1)[:, np.newaxis]
    
    # Calculate and regularize loss
    loss = -np.log(prob[np.arange(N), y])
    loss = np.mean(loss)
    loss = loss + (0.5*reg*np.sum(W*W))
    
    # Calculate score differences for gradient
    dscores = prob
    dscores[np.arange(N), y] = dscores[np.arange(N), y] - 1
    dscores = dscores/N
    
    dW = np.dot(X.T, dscores)
    dW = dW + (reg*W)
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
