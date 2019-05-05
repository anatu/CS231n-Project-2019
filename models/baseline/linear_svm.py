from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # Calculate gradient dW for positive margin
                # Decrement weight on predicted class
                dW[:, y[i]] -= X[i,:]
                # Increment weight for jth class (true class)
                dW[:, j] += X[i,:]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    # Normalize the derivative across number of training examples
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Regularize the gradient
    dW += reg*W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # IMPLEMENTED INLINE - SEE ABOVE!
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = np.arange(X.shape[0])
    # Compute the entire score matrix
    scores = np.dot(X, W)
    # Pull actual class labels for X
    correct_class_scores = scores[N, y]
    # Compute the margin for all scores against actual labels
    margin = np.maximum(0, 1 + (scores - correct_class_scores[:, np.newaxis]))
    # Do not count y_i in the margin
    margin[N, y] = 0
    # Take sum across training samples
    marginSum = np.sum(margin, axis = 1)
    # Average across training samples and regularize loss
    loss = np.mean(marginSum) + 0.5*reg*np.sum(W*W)
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Convert to zero-one loss (gradient of hinge loss) and sum across all training samples
    binaryMargins = margin
    binaryMargins[margin>0] = 1
    binaryMargins[N, y] = -np.sum(binaryMargins, axis = 1).T
    # Compute gradient, normalize, and regularize
    dW = np.dot(X.T, binaryMargins)
    dW = dW / X.shape[0]
    dW = dW + 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
