#!/usr/bin/env/ python
# ECBM E4040 Fall 2023 Assignment 2
# This script contains forward/backward functions of regularization techniques

import numpy as np


def bn_forward(x, gamma, beta, bn_params, mode):
    """
    Batch Normalization forward
    
    The input x has shape (N, D) and contains a minibatch of N
    examples, where each example x[i] has D features. We will apply
    mini-batch normalization on N samples in x. 
    
    In the "train" mode:
    1. Apply normalization transform to input x and store in out.
    2. Update mean and variance estimation in bn_config using moving average method, ie.,
    
    Side note:
    Here we use the moving average strategy to estimiate the mean and var of the data.
    It is kind of approximation to the mean and var of the training data. Also, this is
    a popular strategy and tensorflow use it in their implementation.
    
    In the "test" mode: 
    Instead of using the mean and var of the input data, we are is going to use mean and var
    stored in bn_config to make normalization transform.
    
    :param x: a tensor with shape (N, D)
    :param gamma: (tensor) a scale tensor of length D, a trainable parameter in batch normalization.
    :param beta:  (tensor) an offset tensor of length D, a trainable parameter in batch normalization.
    :param bn_params:  (dict) including epsilon, decay, moving_mean, moving_var.
    :param mode:  (string) "train" or "test".
    
    :return:
    - out: a tensor with the same shape as input x.
    - cahce: (tuple) contains (x, gamma, beta, eps, mean, var)
    """
    eps = bn_params.get("epsilon", 1e-5)
    decay = bn_params.get("decay", 0.9)

    N, D = x.shape
    moving_mean = bn_params.get('moving_mean', np.zeros(D, dtype=x.dtype))
    moving_var = bn_params.get('moving_var', np.ones(D, dtype=x.dtype))

    out = np.zeros_like(x)
    mean = moving_mean
    var = moving_var
    

    if mode == "train":
        #############################################################
        # TODO: Batch normalization forward train mode              #
        #      1. calculate sample mean and sample variance of x         #
        #      2. normalize x with sample mean and sample variance  #
        #      3. transform x to desired distribution (gamma, beta) #
        #      4. estimate moving mean and moving variance          #
        # NOTE: Don't forget to store the parameters!               #
        #############################################################
        #raise NotImplementedError
        # 1. Calculate the sample mean and variance of x
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
    
        # 2. Normalize x using the sample mean and variance
        x_normalized = (x - mean) / np.sqrt(var + eps)

        # 3. Transform the normalized x using gamma and beta
        out = gamma * x_normalized + beta

        # 4. Update the moving mean and variance
        moving_mean = decay * moving_mean + (1 - decay) * mean
        moving_var = decay * moving_var + (1 - decay) * var
        
        # Update bn_params with the new moving mean and variance
        bn_params["moving_mean"] = moving_mean
        bn_params["moving_var"] = moving_var

        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

    elif mode == 'test':
        #############################################################
        # TODO: Batch normalization forward test mode               #
        #       normalize x with moving mean and moving variance    #
        #############################################################
        #raise NotImplementedError
        # Normalize x using the moving mean and variance
        x_normalized = (x - moving_mean) / np.sqrt(moving_var + eps)
        out = gamma * x_normalized + beta
        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

    # cache other intermediate variables for back-propagation
    cache = (x, gamma, beta, eps, mean, var)

    return out, cache


def bn_backward(dout, cache):
    """
    Batch normalization backward
    Derive the gradients wrt gamma, beta and x

    :param dout:  a tensor with shape (N, D)
    :param cache:  (tuple) contains (x, gamma, beta, eps, mean, var)
    
    :return:
    - dx, dgamma, dbeta
    """
    x, gamma, beta, eps, mean, var = cache
    N, D = dout.shape

    x_hat = (x - mean) / np.sqrt(var + eps)

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dout * gamma / np.sqrt(var + eps)

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_config, mode):
    """
    Dropout feedforward
    :param x: input tensor with shape (N, D)
    :param dropout_config: (dict)
                           enabled: (bool) indicate whether dropout is used.
                           keep_prob: (float) retention rate, usually range from 0.5 to 1.
    :param mode: (string) "train" or "test"
    :return:
    - out: a tensor with the same shape as x
    - cache: (train phase) cache a random dropout mask used in feedforward process
             (test phase) None
    """
    keep_prob = dropout_config.get("keep_prob", 1.0)

    out, cache = None, None
    if mode == "train":
        ###########################################
        # TODO: Implement training phase dropout. #
        # Remember to return retention mask for   #
        # backward.                               #
        ###########################################
        #raise NotImplementedError
        pass
        ###########################################
        #             END OF YOUR CODE            #
        ###########################################

    elif mode == "test":
        ##########################################
        # TODO: Implement test phase dropout.    #
        # No need to use mask here.              #
        ##########################################
        #raise NotImplementedError
        pass
        ###########################################
        #             END OF YOUR CODE            #
        ###########################################

    return out, cache


def dropout_backward(dout, cache):
    """
    Dropout backward only for train phase.
    :param dout: a tensor with shape (N, D)
    :param cache: (tensor) mask, a tensor with the same shape as x
    :return: dx: the gradients transfering to the previous layer
    """

    dx = cache * dout
    return dx

