#!/usr/bin/env/ python
# ECBM E4040 Fall 2023 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
import os

try:
    from scipy.ndimage.interpolation import rotate
except ModuleNotFoundError:
    os.system('pip install scipy')
    from scipy.ndimage.interpolation import rotate

class ImageGenerator(object):
    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """
        #######################################################################
        # TODO: Your ImageGenerator instance must store the following:        #
        #           x, y, num_of_samples, height, width, number of pixels,    #
        #           translated, degree of rotation, is_horizontal_flip,       #
        #           is_vertical_flip, is_add_noise.                           #
        #       By default, set boolean values to False.                      #
        #                                                                     #
        # Hint: Since you may directly perform transformations on x and y,    #
        #       and don't want your original data to be contaminated by       #
        #       those transformations, you should use numpy array's           #
        #       build-in copy() method.                                       #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        self.x = x.copy()  # Use numpy's copy method to avoid contamination of original data
        self.y = y.copy()
        self.N = x.shape[0]
        self.height = x.shape[1]
        self.width = x.shape[2]
        self.channel = x.shape[3]
        self.num_pixels = self.height * self.width * self.channel
        self.translated = False
        self.degree_of_rotation = 0
        self.is_horizontal_flip = False
        self.is_vertical_flip = False
        self.is_add_noise = False
        
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################

        # One way to use augmented data is to store them after transformation
        # (and then combine all of them to form new data set)
        # Following variables (along with create_aug_data() function) is one
        # kind of implementation. You can either figure out how to use them or
        # find out your own ways to create the augmented dataset.
        
        # If you have
        # your own idea of creating augmented dataset, just feel free to comment
        # any codes you don't need
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.bright = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.N 
    
    
    def create_aug_data(self):
        # If you want to use function create_aug_data() to generate new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1.store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2.increase self.N_aug by the number of transformed data,
        # 3.you should also return the transformed data in order to show them in task4 notebook
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))
        if self.bright:
            self.x_aug = np.vstack((self.x_aug,self.bright[0]))
            self.y_aug = np.hstack((self.y_aug,self.bright[1]))
            
        print("Size of training data:{}".format(self.N_aug))
        
    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        #######################################################################
        # TODO: Use 'yield' keyword, implement this generator.                #
        #       Pay attention to the following:                               #
        #       1. The generator should return batches endlessly.             #
        #       2. Make sure the shuffle only happens after each sample has   #
        #          been visited once. Else some samples might not appear.     #
        #                                                                     #
        #---------------------------------------------------------------------#
        # One possible pseudo code for your reference:                        #
        #---------------------------------------------------------------------#
        #   calculate the total number of batches possible                    #
        #   (if the rest is not sufficient to make up a batch, ignore)        #
        #   while True:                                                       #
        #       if (batch_count < total number of batches possible):          #
        #           batch_count = batch_count + 1                             #
        #           yield(next batch of x and y indicated by batch_count)     #
        #       else:                                                         #
        #           shuffle(x)                                                #
        #           reset batch_count                                         #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        num_batches = self.N_aug // batch_size
        batch_count = 0
        indices = np.arange(self.N_aug)

        while True:
            if batch_count < num_batches:
                batch_indices = indices[batch_count * batch_size:(batch_count + 1) * batch_size]
                batch_count += 1
                yield self.x_aug[batch_indices], self.y_aug[batch_indices]
            else:
                if shuffle:
                    np.random.shuffle(indices)
                batch_count = 0


    def show(self, images):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        if images.ndim == 3:
        # If images are grayscale, add a channel dimension for plotting
            images = images[..., np.newaxis]

        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i+1)
        # Check if there's a channel dimension and plot accordingly
            if images.shape[-1] == 1:
                plt.imshow(images[i, :, :, 0], cmap='gray')
            else:
                plt.imshow(images[i, :, :, :])
            plt.axis('off')
        plt.show()
        
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """
        #######################################################################
        # TODO: Implement the translate() function. You may wonder what       #
        #       values to append to the edge after the shift. Here, use       #
        #      rolling instead. For example, if you shift 3 pixels to the     #
        #      left, append the left-most 3 columns that are out of boundary  #
        #      to the right edge of the picture.                              #
        #                                                                     #
        # HINT: use np.roll                                                   #
        # https://numpy.org/doc/stable/reference/generated/numpy.roll.html    #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        translated = np.roll(self.x, shift=shift_height, axis=1)
        translated = np.roll(translated, shift=shift_width, axis=2)
        self.translated = (translated, self.y.copy())
        self.N_aug += self.N
        return translated
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################


    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        """      
        self.dor = angle
        rotated = rotate(self.x.copy(), angle,reshape=False,axes=(1, 2))
        print('Currrent rotation: ', self.dor)
        self.rotated = (rotated, self.y.copy())
        self.N_aug += self.N
        return rotated
    

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified.
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        #######################################################################
        # TODO: Implement the flip() function to flip self.x as per the mode. #
        # Mode can be 'h' for horizontal, 'v' for vertical, or 'hv' for both. #
        # HINT: Use numpy's flip function.                                    #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        if mode == 'h':
            flipped = np.flip(self.x, axis=2)  # Horizontal flip
        elif mode == 'v':
            flipped = np.flip(self.x, axis=1)  # Vertical flip
        elif mode == 'hv':
            flipped = np.flip(self.x, axis=(1, 2))  # Both horizontal and vertical flip
        else:
            raise ValueError("Mode should be 'h', 'v', or 'hv'.")
        
        self.flipped = (flipped, self.y.copy())
        self.N_aug += self.N
        return flipped
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################

    
    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """

        assert portion <= 1
        if not self.is_add_noise:
            self.is_add_noise = True
        m = self.N * portion
        index = np.random.choice(self.N, m, replace=False)
        added = self.x.copy()
        for i in index:
            added[i, :, :, :] += np.random.randint(0, 5, [self.height, self.width, self.channel], dtype='uint8') * amplitude
        self.added = (added, self.y.copy()[index])
        self.N_aug += m
        return added


    def brightness(self, factor):
        """
        Scale the pixel values to increase the brightness.
        :param factor: A factor (>=1) by which each pixel in the image will be scaled. 
                    For instance, if the factor is 2, all pixel values will be doubled.
        :return bright: dataset with increased brightness
        """
        #######################################################################
        # TODO: Implement the brightness() function to increase the brightness#
        # of self.x by a given factor. Ensure no pixel value exceeds 255.     #
        #######################################################################
        #                         TODO: YOUR CODE HERE                        #
        #######################################################################
        # raise NotImplementedError
        if factor < 1:
            raise ValueError("Factor must be greater than or equal to 1.")
        
        # Ensure no pixel value exceeds 255 after scaling
        bright = np.clip(self.x * factor, 0, 255).astype(np.uint8)
        
        self.bright = (bright, self.y.copy())
        self.N_aug += self.N
        return bright
        
        #######################################################################
        #                                END TODO                             #
        #######################################################################

      