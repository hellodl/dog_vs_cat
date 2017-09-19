from keras import backend as K
K.set_image_dim_ordering('th')

import keras
from keras.models import Sequential

from keras import layers as keras_layers
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers import convolutional
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import SGD, RMSprop, Adam, Nadam, Adadelta, Adamax, Adagrad
from keras.preprocessing import image
from keras.callbacks import Callback, TensorBoard, LearningRateScheduler, ModelCheckpoint
from keras.models import save_model, load_model
from keras.layers.normalization import BatchNormalization

import os
import numpy as np
import bcolz
import time
from math import ceil
import json
import sys


vgg16_layers = {  # begin_layers
        'Conv2': 6,
        'Conv3': 11,
        'Conv4': 18,
        'Conv5': 25,
        'FC_block': 32,
        'FC_last': -1
    }

def onehot(x):
    return to_categorical(x)

def load_array(fname):
    return bcolz.open(fname)[:]

def get_batch(path, target_size, class_mode, batch_size, gen=None, shuffle=True):
    if gen is None:
        gen = image.ImageDataGenerator()

    return gen.flow_from_directory(path,
                                   target_size=target_size,
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)

def save_array(fname, arr):
    c = bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def proc_wgts(w, p_prev, p_now):
    try:
        f = (1 - p_prev) / (1 - p_now)
    except ZeroDivisionError:
        f = (1 - p_prev) / 0.0001

    return [o * f for o in w]

def vgg_preprocess(x):
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
    x = x - vgg_mean
    return x[:, ::-1]

def ConvBlock(model, nb_block, nb_layer, nb_filter, activation):
    for i in range(nb_block):
        for j in range(nb_layer[i]):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(nb_filter[i], (3, 3), activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

def FCBlock(model, classes, nb_neuron, dropout, activation):
    model.add(Flatten(input_shape=model.layers[-1].output_shape[1:], name='flatten_0'))
    for i in range(len(nb_neuron)):
        model.add(Dense(nb_neuron[i], activation=activation[i], name='dense_%s' % i))
        model.add(Dropout(dropout[i], name='dropout_%s' % i))

    model.add(Dense(classes, activation='softmax', name='dense_out'))

def dropout_adjust(model, dropout_cfg):
    model_dropout_list = []

    for i, l in enumerate(model.layers):
        if type(l) is Dropout:
            model_dropout_list.append([i, l.get_config()['rate']])

    # set new dropout rates:
    for i, l in enumerate(model_dropout_list):
        rate_cfg = dropout_cfg[i]
        l_dropout = model.layers[model_dropout_list[i][0]]
        l_dropout_preLayer = model.layers[model_dropout_list[i][0] - 1]
        rate_pre = l_dropout.get_config()['rate']
        if rate_cfg != rate_pre:
            l_dropout.rate = min(1., max(0., rate_cfg))
            l_dropout_preLayer.set_weights(proc_wgts(l_dropout_preLayer.get_weights(),
                                                     p_prev=rate_pre,
                                                     p_now=rate_cfg))

def parsing_cmd():
    opt = {
        's': 'SGD',
        'r': 'RMSprop',
        'a': 'Adam',
        'n': 'Nadam'
    }
    arg_cnt = len(sys.argv) - 1  # do not count filename
    if arg_cnt == 0:
        return
    opt_sel = None
    ep = None
    lr = 'default'
    for i in sys.argv[1:]:
        if i.startswith('--'):
            cmd_split = i[2:].split('-')
            if cmd_split[0] in opt:
                opt_sel = opt[cmd_split[0]]
                try:
                    lr = min(0.1, float(cmd_split[1]))
                except ValueError:
                    lr = 'default'

            elif cmd_split[0] == 'e':
                try:
                    ep = int(cmd_split[1])
                except IndexError or ValueError:
                    ep = 40
                    print('None integer input for ep, it is set to 40 as default.')

    if opt_sel == None:
        opt_sel = opt['s']

    return opt_sel, lr, ep


def load_base():
    vgg16 = load_model('./model/vgg16_0_2.hdf5')
    vgg16.pop()
    vgg16.pop()
    feat_model = Sequential(vgg16.layers)


def collecting():
    img_size = [224, 224]
    feature_path = './data/feature/'
    trn_features_filename = feature_path + 'trn_ll_feat' + '.bc'
    trn_labels_filename = feature_path + 'trnLabels' + '.bc'
    val_features_filename = feature_path + 'val_ll_feat' + '.bc'
    val_labels_filename = feature_path + 'valLabels' + '.bc'
    tst_features_filename = feature_path + 'tst_ll_feat' + '.bc'
    tst_id_filename = feature_path + 'tstID' + '.bc'

    trn_path = './data/train/'
    val_path = './data/valid/'
    tst_path = './data/test/'

    trn_bsize = 64
    val_bsize = 64
    tst_bsize = 64

    try:
        trn_features = load_array(trn_features_filename)
        trn_labels = load_array(trn_labels_filename)
    except FileNotFoundError:
        trn_batches = get_batch(trn_path,
                                target_size=img_size,
                                class_mode='categorical',
                                shuffle=False,
                                batch_size=trn_bsize)
        trn_labels = onehot(trn_batches.classes)
        trn_features = feat_model.predict_generator(trn_batches,
                                                    ceil(trn_batches.samples / trn_bsize),
                                                    verbose=1)
        save_array(trn_features_filename, trn_features)
        save_array(trn_labels_filename, trn_labels)

    try:
        val_features = load_array(val_features_filename)
        val_labels = load_array(val_labels_filename)
    except FileNotFoundError:
        val_batches = get_batch(val_path,
                                target_size=img_size,
                                class_mode='categorical',
                                shuffle=False,
                                batch_size=val_bsize)
        val_labels = onehot(val_batches.classes)
        val_features = feat_model.predict_generator(val_batches,
                                                    ceil(val_batches.samples / val_bsize),
                                                    verbose=1)
        save_array(val_features_filename, val_features)
        save_array(val_labels_filename, val_labels)

    try:
        tst_features = load_array(tst_features_filename)
        tst_ids = load_array(tst_id_filename)
    except FileNotFoundError:
        test_batches = get_batch(tst_path,
                                 target_size=img_size,
                                 class_mode='categorical',
                                 shuffle=False,
                                 batch_size=tst_bsize)

        tst_ids = np.array([int(f[8:f.find('.')]) for f in test_batches.filenames])
        tst_features = feat_model.predict_generator(test_batches, ceil(test_batches.samples / tst_bsize),
                                                    verbose=1)

        save_array(tst_features_filename, tst_features)
        save_array(tst_id_filename, tst_ids)

    return [trn_features, trn_labels], [val_features, val_labels]


def load_ll_model(opt):
    def get_ll_layers():
        return [
            BatchNormalization(input_shape=(4096,)),
            Dropout(0.60),
            Dense(2, activation='softmax')
        ]

    ll_layers = get_ll_layers()
    ll_model = Sequential(ll_layers)
    ll_model.compile(optimizer=eval(opt)(), loss='categorical_crossentropy', metrics=['accuracy'])
    return ll_model


def fit(model, lr, ep, t_data, v_data):
    model.optimizer.lr = lr
    model.fit(t_data[0], t_data[1],
             validation_data=(v_data[0], v_data[1]),
             batch_size=64,
             epochs=ep)


if __name__ == '__main__':
    opt, lr, ep = parsing_cmd()
    print(opt+'-'+lr+'-'+ep)
    feat_model = load_base()
    ll_model = load_ll_model(opt)
    t_data, v_data = collecting()
    fit(ll_model, lr, ep, t_data, v_data)
