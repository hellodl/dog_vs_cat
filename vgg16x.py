import numpy as np
import os, json

import keras
from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D,Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.preprocessing import image
from keras import backend as K
from math import ceil

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr

class Vgg16():
    """
        The vgg16 model
    """
    def __init__(self):
        self.create()
        self.get_classes()
        print('vgg16 is initialized.')

    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache
        :return: 
        """

        with open('imagenet_class_index.json') as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model
        :param imgs: 
        :param details: 
        :return: 
        """
        # predict probability of each class for each image
        all_preds = self.model.predict(imgs)
        # for each image get the index of the class with max probability
        idxs = np.argmax(all_preds, axis=1)
        # get the values of the highest probability for each image
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        # get the label of the class with the highest probability for each image
        classes = [self.classes[i] for i in idxs]
        return np.array(preds), idxs, classes

    def create(self):
        model = self.model = Sequential()

        model.add(
            Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))  # notice the in/out_shape

        print(model.output_shape)
        self.ConvBlock(2, 64)
        print(model.output_shape)
        self.ConvBlock(2, 128)
        print(model.output_shape)
        self.ConvBlock(3, 256)
        print(model.output_shape)
        self.ConvBlock(3, 512)
        print(model.output_shape)
        self.ConvBlock(3, 512)
        print(model.output_shape)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        # fname = 'vgg16.h5'
        # model.load_weights(get_file(fname, fname))
        model.load_weights('vgg16.h5')

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224, 224),
                                       class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def ConvBlock(self, layers, filters):
        """

        :param layers: 
            (int) The number of zero padded convolution layers to be added to the model
        :param filters: 
            (int) The number of convolution filters to be created for each layer.
        :return: 
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a Dropout of 0.5
        :return: 
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

    def ft(self, num):
        """
            replace the last layer of the model with a Dense(fully connected) layer of num neurons.
            will also lock the weights of all layers except the new layer so that we only learn weigths
            for the last layer in subsequent training
        :param num: 
            Number of neurons in the Dense layer
        :return: 
            None
        """
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(num, activation='softmax'))
        K.set_image_dim_ordering('th')
        self.compile()

    def compile(self, lr=0.01):
        self.model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_data(self, trn, labels, val, val_labels, nb_epoch=1, batch_size=64):
        """
            Trains the model for a fixed number of epochs(iterations on a dataset).
            See Keras documentation: http://keras.io/models/model/

        :param trn: 
        :param labels: 
        :param val: 
        :param val_labels: 
        :param nb_epoch: 
        :param batch_size: 
        :return: 
        """
        self.model.fit(trn, labels, nb_epoch=nb_epoch, validation_data=(val, val_labels), batch_size=batch_size)

    def fit(self, batches, val_batches, nb_epoch=1, verbose=1):
        """
            Fits the model on data yielded batch-by-batch by a python generator.

        :param batches: 
        :param val_batches: 
        :param nb_epoch: 
        :return: 
        """
        # self.model.fit_generator(batches, samples_per_epoch=batches.samples, nb_epoch=nb_epoch,
        #                         validation_data=val_batches, nb_val_samples=val_batches.samples)

        print(batches.samples)
        self.model.fit_generator(batches, steps_per_epoch=ceil(batches.samples / batches.batch_size), validation_data=val_batches,
                                 epochs=nb_epoch,
                                 validation_steps=ceil(val_batches.samples / val_batches.batch_size), verbose= verbose)

    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.

        :param batches: 
        :return: 
        """
        self.ft(batches.num_class)
        classes = list(iter(batches.class_indices))

        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes

    def test(self, batches, verbose=1):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.

            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.

        """
        return self.model.predict_generator(batches, steps=ceil(batches.samples / batches.batch_size), verbose=verbose)
