#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE
    n = x.shape[0]
    x /= np.sqrt(np.sum(x**2, axis=1)).reshape((n,1))
    ### END YOUR CODE
    return x

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE
    ndim = x.ndim
    Max_Value_Row = np.max(x, axis = ndim - 1)
    numerator = np.exp(x - Max_Value_Row)
    denominator = np.sum(numerator, axis = ndim - 1)
    x = numerator / denominator
    ### END YOUR CODE
    return x