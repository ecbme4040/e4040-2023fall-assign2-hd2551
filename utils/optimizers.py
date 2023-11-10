#!/usr/bin/env/ python
# ECBM E4040 Fall 2023 Assignment 2
# Optimizer implementations

import numpy as np


class Optimizer:
    def __init__(self, model):
        """
        :param model: (class MLP) an MLP model
        """
        self.model = model

    def zero_grad(self):
        """
        Clear all past gradients for the next update
        """

        grads = self.model.grads
        # for each parameter
        for k in grads:
            grads[k] = 0

    def train(
        self, X_train, y_train, X_valid, y_valid,
        num_epoch=10, batch_size=500, learning_rate=1e-3,
        learning_decay=0.95, verbose=False, record_interval=10
    ):
        """
        This function is for MLP model training

        Inputs:
        :param X_train: (float32) input data, a tensor with shape (N, D1, D2, ...)
        :param y_train: (int) label data for classification, a 1D array of length N
        :param X_valid: (float32) input data, a tensor with shape (num_valid, D1, D2, ...)
        :param y_valid: (int) label data for classification, a 1D array of length num_valid
        :param num_epoch: (int) the number of training epochs
        :param batch_size: (int) the size of a single batch for training
        :param learning_rate: (float)
        :param learning_decay: (float) a factor for reducing the learning rate in every epoch
        :param stochastic: (boolean) whether to use stochastic gradient
        """
        model = self.model
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        print('number of batches for training: {}'.format(num_batch))

        # recorder
        loss_hist = []
        train_acc_hist = []
        valid_acc_hist = []
        loss = 0.0

        # loop
        for e in range(num_epoch):
            # train stage
            model.cnter = 0

            for i in range(num_batch):
                # batch
                X_batch = X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = y_train[i * batch_size:(i + 1) * batch_size]

                # clear gradients before each batch
                self.zero_grad()
                self.X_batch = X_batch
                self.y_batch = y_batch

                # forward
                preds = model.forward(X_batch)
                # loss
                loss += model.loss(preds, y_batch)

                # update gradients after each batch
                self.step(learning_rate=learning_rate)

                if (i + 1) % record_interval == 0:
                    loss /= record_interval
                    loss_hist.append(loss)
                    if verbose:
                        print('{}/{} loss: {}'.format(batch_size * (i + 1), num_train, loss))
                    loss = 0.0

            # validation stage
            train_acc = model.check_accuracy(X_train, y_train)
            val_acc = model.check_accuracy(X_valid, y_valid)
            train_acc_hist.append(train_acc)
            valid_acc_hist.append(val_acc)

            # Shrink learning_rate
            learning_rate *= learning_decay
            print(
                'epoch {}: valid acc = {}, new learning rate = {}, '
                'number of evaluations {}'.format(e + 1, val_acc, learning_rate, model.cnter)
            )

        return loss_hist, train_acc_hist, valid_acc_hist

    def test(self, X_test, y_test, batch_size=10000):
        """
        Inputs:
        :param X_test: (float) a tensor of shape (N, D1, D2, ...)
        :param y_test: (int) an array of length N
        :param batch_size: (int) seperate input data into several batches
        """
        model = self.model
        acc = 0.0
        num_test = X_test.shape[0]

        if num_test <= batch_size:
            acc = model.check_accuracy(X_test, y_test)
            print('accuracy in a small test set: {}'.format(acc))
            return acc

        num_batch = num_test // batch_size
        for i in range(num_batch):
            X_batch = X_test[i * batch_size:(i + 1) * batch_size]
            y_batch = y_test[i * batch_size:(i + 1) * batch_size]
            acc += batch_size * model.check_accuracy(X_batch, y_batch)

        X_batch = X_test[num_batch * batch_size:]
        y_batch = y_test[num_batch * batch_size:]
        if X_batch.shape[0] > 0:
            acc += X_batch.shape[0] * model.check_accuracy(X_batch, y_batch)

        acc /= num_test
        print('test accuracy: {}'.format(acc))
        return acc

    def step(self, learning_rate):
        """
        For the subclasses to implement
        """
        raise NotImplementedError


class SGDOptim(Optimizer):
    def step(self, learning_rate):
        """
        Implement SGD update on network parameters
        
        Inputs:
        :param learning_rate: (float)
        """
        # get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads

        # update each parameter
        for k in grads:
            params[k] -= learning_rate * grads[k]


class SGDMomentumOptim(Optimizer):
    def __init__(self, model, momentum=0.5):
        """
        Inputs:

        :param model: a neural netowrk class object
        :param momentum: (float) momentum decay factor
        """
        super().__init__(model)
        self.momentum = momentum
        velocities = dict()
        for k, v in model.params.items():
            velocities[k] = np.zeros_like(v)
        self.velocities = velocities

    def step(self, learning_rate):
        """
        Implement a one-step SGD + Momentum update on network's parameters
        
        Inputs:
        :param learning_rate: (float)
        """
        momentum = self.momentum
        velocities = self.velocities

        # get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads
        ###################################################
        # TODO: SGD+Momentum, Update params and velocities#
        ###################################################
        #raise NotImplementedError
        
        # Update velocities and parameters
        for k in grads:
            # Update velocity
            velocities[k] = momentum * velocities[k] + (1 - momentum) * grads[k]
            # Update parameters
            params[k] -= learning_rate * velocities[k]
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################



class SGDNestMomentumOptim(SGDMomentumOptim):
    def step(self, learning_rate):
        """
        Implement a one-step SGD + Nesterov Momentum update on network's parameters
        
        Inputs:
        :param learning_rate: (float)
        """
        momentum = self.momentum
        velocities = self.velocities

        # get all parameters and their gradients
        params = self.model.params
        grads = self.model.grads
        ###################################################
        # TODO: SGD+Momentum, Update params and velocities#
        ###################################################
        #raise NotImplementedError
        for k in grads:
            
            interim_param = params[k] + momentum * velocities[k]
            
            # Compute gradient at interim parameters
            interim_grad = grads[k]  
            
            # Update velocity using interim gradient
            velocities[k] = momentum * velocities[k] - learning_rate * interim_grad
            
            # Update parameters using the updated velocity
            params[k] += velocities[k]
        
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################


class AdamOptim(Optimizer):
    def __init__(self, model, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Inputs:
        :param model: a neural network class object
        :param beta1: (float) should be close to 1
        :param beta2: (float) similar to beta1
        :param eps: (float) in different case, the good value for eps will be different
        """
        super().__init__(model)

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # initialize m0 and v0
        self.mean, self.var = {}, {}
        for k, v in model.params.items():
            self.mean[k] = np.zeros_like(v)
            self.var[k] = np.zeros_like(v)

        # time step
        self.t = 0

    def step(self, learning_rate):
        """
        Implement a one-step Adam update on network's parameters

        Inputs:
        :param learning_rate: (float)
        """
        # update time step
        self.t += 1

        # get params
        model = self.model
        beta1 = self.beta1
        beta2 = self.beta2
        eps = self.eps
        t = self.t

        # stored moments
        mean = self.mean
        var = self.var

        # get all parameters and their gradients
        params = model.params
        grads = model.grads
        ###################################################
        # TODO: Adam, Update t, moments and params        #
        ###################################################
        #raise NotImplementedError
        for k in grads:
            # Update biased first moment estimate
            mean[k] = beta1 * mean[k] + (1 - beta1) * grads[k]
            
            # Update biased second raw moment estimate
            var[k] = beta2 * var[k] + (1 - beta2) * (grads[k] ** 2)
            
            # Compute bias-corrected first moment estimate
            mean_corrected = mean[k] / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            var_corrected = var[k] / (1 - beta2 ** t)
            
            # Update parameters
            params[k] -= learning_rate * mean_corrected /((np.sqrt(var_corrected) + eps)) 
        ###################################################
        #               END OF YOUR CODE                  #
        ###################################################


class BacktraceSGDOptim(SGDOptim):
    '''
    This class is derived from SGDmomentumOptim. 
    We will use the same SGDmomentumOptim.step function for updates. 
    Line search will be implemented by rewriting the SGDmomentumOptim.train function. 

    Please read through the code and follow the instructions below.
    '''

    def __init__(self, model, momentum=0.5, c=1, beta=0.5, max_searches=10):
        """
        Inputs:
        :param model: a neural network class object
        :param momentum: (float) momentum decay factor
        :param c: (float) line search upper-bound slope
        :param beta: (float) line search learning rate decay factor
        :param max_searches: (int) maximum number of search loops in a single iteration
        :param eps: (float) in different case, the good value for eps will be different
        """
        super().__init__(model)

        self.c = c
        self.beta = beta
        self.max_searches = max_searches

    def step(self, learning_rate):
        # get params
        model = self.model
        c = self.c
        beta = self.beta
        max_searches = self.max_searches

        # data
        X_batch = self.X_batch
        y_batch = self.y_batch

        # get current loss L_t
        # we do this outside the loop since this quantity is fixed
        preds = model.forward(X_batch)
        loss_curr = model.loss(preds, y_batch)

        # get a copy of all parameters and their gradients
        params = {k: v.copy() for k, v in model.params.items()}
        grads = {k: v.copy() for k, v in model.grads.items()}

        # set initial step size
        alpha = learning_rate

        # another fixed quantity for upper bound calculation
        G = 0
        for k in grads:
            G += c * np.sum(grads[k] * grads[k])

        # search iteration
        for i in range(1, max_searches):
            # reset the parameters at every iteration
            # so that a bad previous attempt won't affect the next
            model.params = params
            model.grads = grads
            
            ###################################################
            # TODO: SGD + Backtrace                           #
            #       Follow the prompts given below.           #
            ###################################################
            #raise NotImplementedError
            
            # shrink the step size
            alpha *= beta

            # compute the Armijo upper bound
            upper_bound = loss_curr - alpha * G

            # update model with the current step size
            # you may directly call super().step(current_stepsize)
            super().step(alpha)


            # compute next loss L_{t+1}
            preds_next = model.forward(X_batch)
            loss_next = model.loss(preds_next, y_batch)

            # re-evaluate the model on X_batch and y_batch
            preds_next = model.forward(X_batch)
            loss_next = model.loss(preds_next, y_batch)

            # stopping criterion
            # compare L_{t+1} with the upper bound
            if loss_next <= upper_bound or alpha < 1e-5: 
                break

            ###################################################
            #               END OF YOUR CODE                  #
            ###################################################

        # print the number of search steps taken
        print('Searched {} times, final step size = {}'.format(i, learning_rate), end='\r')
