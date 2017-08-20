from input_timeout import input_to
from imp import reload
import utils
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.preprocessing import image
import vgg16x
import math
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.pooling import MaxPooling2D
import time
import math
import numpy as np


def proc_wgts(layer): return [o/2 for o in layer.get_weights()]


def get_fc_model(opt, conv_layers, fc_layers):
    model = Sequential([
        MaxPooling2D(input_shape=conv_layers[-1].output_shape[1:]),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(4096, activation='relu'),
        Dropout(0.),
        Dense(2, activation='softmax')
        ])

    for l1,l2 in zip(model.layers, fc_layers): l1.set_weights(proc_wgts(l2))

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def op_0(nb_ep, saveWeights=True, modelTest=True, aug=False):
    K.set_image_dim_ordering('th')
    reload(vgg16x);reload(utils)

    model = utils.vgg_ft(2)
    model_path = './results_redux/'
    model_name = 'ft7_17-08-02_22:50:23.h5'
    model.load_weights(model_path + model_name)

    layers = model.layers
    last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Conv2D][-1]

    opt = RMSprop(lr=0.00001, rho=0.7)
    conv_layers = layers[:last_conv_idx + 1]
    conv_model = Sequential(conv_layers)
    fc_layers = layers[last_conv_idx + 1:]

    fc_model = get_fc_model(opt=opt, conv_layers=conv_layers, fc_layers=fc_layers)

    for layer in conv_model.layers: layer.trainable = False
    # Look how easy it is to connect two models together!
    conv_model.add(fc_model)
    conv_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # prepare the data
    batch_size = 64
    if aug is True:
        gen = image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                       height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
        batches = utils.get_batches('./train_redux', gen, batch_size=batch_size)
    else:
        batches = utils.get_batches('./train_redux', shuffle=False, batch_size=batch_size)

    val_batches = utils.get_batches('./valid_redux', shuffle=False, batch_size=batch_size)

    val_classes = val_batches.classes
    trn_classes = batches.classes
    val_labels = utils.onehot(val_classes)
    trn_labels = utils.onehot(trn_classes)

    test_batches = utils.get_batches('./test_redux', shuffle=False, batch_size=100, class_mode=None)

    # load features
    trn_features = utils.load_array('train_convlayer_features.bc')
    val_features = utils.load_array('valid_convlayer_features.bc')
    tst_features = utils.load_array('test_convlayer_features.bc')
    print(trn_features.shape)
    print(val_features.shape)
    print(tst_features.shape)

    # fc_model.fit(trn_features, trn_labels, epochs=8,
    #             batch_size=batch_size, validation_data=(val_features, val_labels))

    ISOTIMEFORMAT = '%y-%m-%d_%X'
    latest_weights_filename = None
    result_path = 'results_redux'
    for epoch in range(nb_ep):
        epochs_done = epoch + 1
        print('Running epoch: %d' % epochs_done)

        fc_model.fit(trn_features, trn_labels, epochs=1,
                     batch_size=batch_size, validation_data=(val_features, val_labels))

        curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
        latest_weights_filename = 'ft%d_%s.h5' % (epoch, curTime)
        if saveWeights is True:
            fc_model.save_weights(result_path + '/' +latest_weights_filename)

        if modelTest is True:
            preds = fc_model.predict(tst_features, batch_size=100, verbose=1)
            filenames = test_batches.filenames

            isdog = preds[:, 1]
            isdog = isdog.clip(min=0.05, max=0.95)
            print(isdog[:100])
            print(isdog.shape)
            ids = np.array([int(f[8:f.find('.')]) for f in filenames])
            subm = np.stack([ids, isdog], axis=1)

            ISOTIMEFORMAT = '%y-%m-%d_%X'
            curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
            submission_file_name = 'submission_%s.csv' % curTime
            np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')

if __name__ == '__main__':
    op_0(3)
