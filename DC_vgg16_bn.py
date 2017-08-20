import numpy as np
import os,json,time
import math
import bcolz
from imp import reload

import keras
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint

from input_timeout import input_to
from plt import plotCurve

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def onehot(x):
    return to_categorical(x)

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

class dc_vgg16_bn():

    def __init__(self):
        self.model = None
        self.FC_model = None
        self.opt = None

        self.data_path = os.getcwd()
        self.train_path = 'train_redux'
        self.valid_path = 'valid_redux'
        self.test_path = 'test_redux'
        self.result_path = 'results_redux'
        self.model_path = 'model_redux'
        self.sub_path = 'sub_redux'
        self.ftModel_path = None
        self.train_batches = None
        self.valid_batches = None
        self.test_batches = None
        self.trnBSize = 64
        self.valBSize = 64
        self.tstBSize = 64
        self.tmFm = '%m-%d_%X'

        self.lr = 0.00001

    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache
        :return: 
        """
        with open('imagenet_class_index.json') as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def setTrainPath(self, str):
        self.train_path = self.data_path + '/' + str

    def setValidPath(self, str):
        self.valid_path = self.data_path + '/' + str

    def setTestPath(self, str):
        self.test_path = self.data_path + '/' + str

    def setResultPath(self, str):
        self.result_path = self.data_path + '/' + str

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

    def FCBlock(self, dropoutP=0.5):
        """
            Adds a fully connected layer of 4096 neurons to the model with a Dropout of 0.5
        :return: 
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(dropoutP))

    def baseModel(self, w_file):
        self.get_classes()

        model = self.model = Sequential()

        model.add(
            Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock(dropoutP=0.5)
        self.FCBlock(dropoutP=0.5)
        model.add(Dense(1000, activation='softmax'))

        # fname = 'vgg16.h5'
        # model.load_weights(get_file(fname, fname))
        model.load_weights(w_file)  # 'vgg16.h5'
        print(model.summary())
        input('pause')

    def setOpt(self, str_in, lr):
        if str_in == 'RMSprop':
            self.opt = RMSprop(lr=lr, rho=0.7)
        elif str_in == 'SGD':
            self.opt = SGD()
        elif str_in == 'Nadam':
            self.opt = Nadam(lr=lr)
        else:
            print("No optimizer called '%s'" % str)
        print(self.opt)

    def compile(self):
        if self.opt is not None:
            print('Compiling the model...')
            self.model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            print('No opt is set yet for model compiling.')

    def get_fc_model(self, input_shape, train_layers, fc_layers=None,  dp0=0.5, dp1=0.5):
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(4096, activation='relu'),
            Dropout(dp0),
            Dense(4096, activation='relu'),
            Dropout(dp1),
            Dense(2, activation='softmax')
        ])

        if fc_layers is not None:
            for l1, l2 in zip(model.layers, fc_layers): l1.set_weights(l2.get_weights())

        for layer in model.layers[: (-train_layers)]:
            layer.trainable = False

        return model

    def ftFC_pred(self, test_features, test_ids):
        fc_model = self.FC_model
        fc_model.load_weights(self.ftModel_path)
        preds = fc_model.predict(test_features, batch_size=100, verbose=2)

        isdog = preds[:, 1]
        isdog = isdog.clip(min=0.005, max=0.995)
        subm = np.stack([test_ids, isdog], axis=1)

        curTime = time.strftime(self.tmFm, time.localtime())
        submission_file_name = self.sub_path+ '/' +'sub_%s.csv' % curTime
        np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')

    def ftFC_Last(self, batches, nb_epoch):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.

        :param batches: 
        :return: 
        """
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False
        model.add(Dense(2, activation='softmax'))
        self.compile()

        layers = model.layers
        last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Conv2D][-1]
        conv_layers = layers[:last_conv_idx + 2]
        fc_layers = layers[last_conv_idx + 2:]

        fc_model = self.FC_model = self.get_fc_model(conv_layers[-1].output_shape[1:], fc_layers=fc_layers, train_layers=1)
        fc_model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])


        try:
            trn_features = load_array('trnConvFeatures.bc')
            val_features = load_array('valConvFeatures.bc')
            tst_features = load_array('tstConvFeatures.bc')

            trn_labels = load_array('trnLabels.bc')
            val_labels = load_array('valLabels.bc')
            test_ids = load_array('test_id.bc')

        except FileNotFoundError:
            print("One or more files are not found.")
            conv_model = Sequential(conv_layers)

            trn_batches = self.get_batches('./train_redux', shuffle=False, batch_size=self.trnBSize)
            val_batches = self.get_batches('./valid_redux', shuffle=False, batch_size=self.valBSize)
            test_batches = self.get_batches('./test_redux', shuffle=False, batch_size=self.tstBSize, class_mode=None)
            test_ids = np.array([int(f[8:f.find('.')]) for f in test_batches.filenames])

            trn_labels = onehot(trn_batches.classes)
            val_labels = onehot(val_batches.classes)

            trn_features = conv_model.predict_generator(trn_batches, math.ceil(trn_batches.samples / self.trnBSize), verbose=1)
            val_features = conv_model.predict_generator(val_batches, math.ceil(val_batches.samples / self.valBSize), verbose=1)
            tst_features = conv_model.predict_generator(test_batches, math.ceil(test_batches.samples / self.tstBSize),
                                                        verbose=1)

            save_array('trnConvFeatures.bc', trn_features)
            save_array('valConvFeatures.bc', val_features)
            save_array('tstConvFeatures.bc', tst_features)

            save_array('trnLabels.bc', trn_labels)
            save_array('valLabels.bc', val_labels)
            save_array('test_id.bc', test_ids)

            classes = list(iter(trn_batches.class_indices))

            # batches.class_indices is a dict with the class name as key and an index as value
            # eg. {'cats': 0, 'dogs': 1}

            for c in trn_batches.class_indices:
                classes[trn_batches.class_indices[c]] = c
            self.classes = classes

        def scheduler(epoch):
            lr = K.get_value(self.model.optimizer.lr)
            print('epoch(%d): here is the lr: %.10f.' % (epoch, lr))
            return float(lr/2)

        lrHist=lrHistory()
        # lrate = LearningRateScheduler(scheduler)
        curTime = time.strftime(self.tmFm, time.localtime())
        self.ftModel_path = self.model_path + '/' + 'ftFC_Last_%s.hdf5' % curTime
        checkpointer = ModelCheckpoint(filepath=self.ftModel_path,
                                       verbose=1,
                                       save_best_only=True)

        hist = fc_model.fit(trn_features, trn_labels, epochs=nb_epoch,
                     batch_size=self.trnBSize, validation_data=(val_features, val_labels), callbacks=[lrHist, checkpointer])
        fitRecords = hist.history
        plotCurve(l_str='loss', l_val=fitRecords['loss'], lr_init=self.lr, r_str='acc', r_val=fitRecords['acc'])

        self.ftFC_pred(test_features=tst_features, test_ids=test_ids)

    def ftFC_Block(self, nb_epoch, fc_w, model_check='val_loss'):
        model = self.model
        layers = model.layers
        last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Conv2D][-1]

        conv_layers = layers[:last_conv_idx + 2]
        fc_layers = layers[last_conv_idx + 2:]
        fc_model = self.FC_model = self.get_fc_model(input_shape=conv_layers[-1].output_shape[1:], train_layers=5)
        fc_model.load_weights(fc_w)
        fc_model.compile(optimizer=self.opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(fc_model.summary())

        input('pause')
        try:
            trn_features = load_array('trnConvFeatures.bc')
            val_features = load_array('valConvFeatures.bc')
            tst_features = load_array('tstConvFeatures.bc')

            trn_labels = load_array('trnLabels.bc')
            val_labels = load_array('valLabels.bc')
            test_ids = load_array('test_id.bc')

        except FileNotFoundError:
            print("One or more files are not found.")
            conv_model = Sequential(conv_layers)
            trn_batches = self.get_batches('./train_redux', shuffle=False, batch_size=self.trnBSize)
            val_batches = self.get_batches('./valid_redux', shuffle=False, batch_size=self.valBSize)
            test_batches = self.get_batches('./test_redux', shuffle=False, batch_size=self.tstBSize, class_mode=None)
            test_ids = np.array([int(f[8:f.find('.')]) for f in test_batches.filenames])

            trn_labels = onehot(trn_batches.classes)
            val_labels = onehot(val_batches.classes)

            trn_features = conv_model.predict_generator(trn_batches, math.ceil(trn_batches.samples / self.trnBSize),
                                                        verbose=1)
            val_features = conv_model.predict_generator(val_batches, math.ceil(val_batches.samples / self.valBSize),
                                                        verbose=1)
            tst_features = conv_model.predict_generator(test_batches, math.ceil(test_batches.samples / self.tstBSize),
                                                        verbose=1)

            save_array('trnConvFeatures.bc', trn_features)
            save_array('valConvFeatures.bc', val_features)
            save_array('tstConvFeatures.bc', tst_features)

            save_array('trnLabels.bc', trn_labels)
            save_array('valLabels.bc', val_labels)
            save_array('test_id.bc', test_ids)

            classes = list(iter(trn_batches.class_indices))

            # batches.class_indices is a dict with the class name as key and an index as value
            # eg. {'cats': 0, 'dogs': 1}

            for c in trn_batches.class_indices:
                classes[trn_batches.class_indices[c]] = c
            self.classes = classes

        def scheduler(epoch):
            lr = K.get_value(self.model.optimizer.lr)
            print('epoch(%d): here is the lr: %.10f.' % (epoch, lr))
            return float(lr / 2)

        lrHist = lrHistory()
        # lrate = LearningRateScheduler(scheduler)
        curTime = time.strftime(self.tmFm, time.localtime())
        self.ftModel_path = self.model_path + '/' + 'ftFC_Block_%s.hdf5' % curTime
        checkpointer = ModelCheckpoint(filepath=self.ftModel_path,
                                       monitor=model_check,
                                       verbose=1,
                                       save_best_only=True)

        hist = fc_model.fit(trn_features, trn_labels, epochs=nb_epoch,
                            batch_size=self.trnBSize, validation_data=(val_features, val_labels), callbacks=[lrHist, checkpointer])
        fitRecords = hist.history
        plotCurve(l_str='loss', l_val=fitRecords['loss'], lr_init=self.lr, r_str='acc', r_val=fitRecords['acc'])

        self.ftFC_pred(test_features=tst_features, test_ids=test_ids)

    def ftFC_Block_Aug(self, nb_epoch, fc_w, conv_feature_file, conv_feature_path, model_check='val_loss'):
        model = self.model
        layers = model.layers
        last_conv_idx = [index for index, layer in enumerate(layers) if type(layer) is Conv2D][-1]

        conv_layers = layers[:last_conv_idx + 2]
        fc_layers = layers[last_conv_idx + 2:]
        fc_model = self.FC_model = self.get_fc_model(input_shape=conv_layers[-1].output_shape[1:], train_layers=5)
        fc_model.load_weights(fc_w)
        fc_model.compile(optimizer=self.opt, loss='binary_crossentropy', metrics=['accuracy'])

        conv_model = Sequential(conv_layers)
        try:
            trn_features = load_array(conv_feature_path + conv_feature_file)
            trn_labels = load_array('trnLabels.bc')

        except FileNotFoundError:
            gen = image.ImageDataGenerator(rotation_range= 20,
                                           shear_range = 0.1,
                                           zoom_range=[0.8, 1.05],
                                           width_shift_range=0.1,
                                           height_shift_range=0.1,
                                           horizontal_flip=True,
                                           fill_mode='constant')

            for i in range(5):
                trn_batches = self.get_batches('./train_redux', gen=gen, shuffle=False, batch_size=self.trnBSize)
                trn_labels = onehot(trn_batches.classes)
                trn_features = conv_model.predict_generator(trn_batches, math.ceil(trn_batches.samples / self.trnBSize),
                                                        verbose=1)

                save_array(conv_feature_path + 'trnConvFeatures_aug%d.bc' %i, trn_features)
                save_array('trnLabels_%d.bc' %i, trn_labels)

                classes = list(iter(trn_batches.class_indices))

            # batches.class_indices is a dict with the class name as key and an index as value
            # eg. {'cats': 0, 'dogs': 1}

                for c in trn_batches.class_indices:
                    classes[trn_batches.class_indices[c]] = c
                self.classes = classes

            input('pause')
        try:
            val_features = load_array('valplus.bc')
            val_labels = load_array('valplus_Labels.bc')

        except FileNotFoundError:
            val_batches = self.get_batches('./valid_plus', shuffle=False, batch_size=self.valBSize)
            val_labels = onehot(val_batches.classes)
            val_features = conv_model.predict_generator(val_batches, math.ceil(val_batches.samples / self.valBSize),
                                                        verbose=1)
            save_array('valplus.bc', val_features)
            save_array('valplus_Labels.bc', val_labels)

        try:
            tst_features = load_array('tstConvFeatures.bc')
            test_ids = load_array('test_id.bc')

        except FileNotFoundError:

            test_batches = self.get_batches('./test_redux', shuffle=False, batch_size=self.tstBSize, class_mode=None)
            test_ids = np.array([int(f[8:f.find('.')]) for f in test_batches.filenames])
            tst_features = conv_model.predict_generator(test_batches, math.ceil(test_batches.samples / self.tstBSize),
                                                        verbose=1)
            save_array('tstConvFeatures.bc', tst_features)
            save_array('test_id.bc', test_ids)


        def scheduler(epoch):
            lr = K.get_value(self.model.optimizer.lr)
            print('epoch(%d): here is the lr: %.10f.' % (epoch, lr))
            return float(lr / 2)

        lrHist = lrHistory()
        sWeights = saveWeights()
        sWeights.set_lossLine(0.0270)
        # lrate = LearningRateScheduler(scheduler)
        curTime = time.strftime(self.tmFm, time.localtime())
        self.ftModel_path = self.model_path + '/' + 'ftFC_Block_%s.hdf5' % curTime
        checkpointer = ModelCheckpoint(filepath=self.ftModel_path,
                                       monitor=model_check,
                                       verbose=1,
                                       save_best_only=True)

        hist = fc_model.fit(trn_features, trn_labels, epochs=nb_epoch,
                            batch_size=self.trnBSize, validation_data=(val_features, val_labels),
                            callbacks=[lrHist, checkpointer])
        fitRecords = hist.history
        # plotCurve(l_str='loss', l_val=fitRecords['loss'], lr_init=self.lr, r_str='acc', r_val=fitRecords['acc'])
        # self.ftModel_path = sWeights.getWeightsName()
        print('Using weight file: %s' %self.ftModel_path)
        cmd = input('Predict?')
        if cmd == 'N' or cmd == 'n':
            return
        self.ftFC_pred(test_features=tst_features, test_ids=test_ids)

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

        self.model.fit_generator(batches, steps_per_epoch=math.ceil(batches.samples / batches.batch_size), validation_data=val_batches,
                                 epochs=nb_epoch,
                                 validation_steps=math.ceil(val_batches.samples / val_batches.batch_size), verbose= verbose)

    def allBatches(self, trnBatch_size=None, valBatch_size=None, tstBatch_size=None):
        trSize = (trnBatch_size if trnBatch_size is not None else self.trnBSize)
        vlSize = (valBatch_size if valBatch_size is not None else self.valBSize)
        tsSize = (tstBatch_size if tstBatch_size is not None else self.tstBSize)
        print('Getting train data with the batch size as %s.' % trSize)
        print('Getting valid data with the batch size as %s.' % vlSize)
        print('Getting test data with the batch size as %s.' % tsSize)
        self.train_batches = self.get_batches(self.train_path, batch_size=trSize)
        self.valid_batches = self.get_batches(self.valid_path, batch_size=vlSize)
        self.test_batches = self.get_batches(self.test_path, shuffle=False, batch_size=tsSize, class_mode=None)

    def trainLastFC(self, opt, lr, nb_epoch, auto_save=True):
        # self.allBatches()
        # self.setOpt('RMSprop', self.lr)
        self.lr = lr
        self.setOpt(opt,self.lr)
        self.ftFC_Last(self.train_batches, nb_epoch=nb_epoch)

        # ISOTIMEFORMAT = '%d_%X'
        # curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
        # print(curTime)

    def trainFCBlock(self, opt, lr, nb_epoch, fc_w, model_check='val_loss', auto_save=True):
        self.lr = lr
        self.setOpt(opt, self.lr)
        self.ftFC_Block(nb_epoch=nb_epoch, fc_w=self.model_path+'/'+fc_w, model_check=model_check)

    def trainFCBlock_Aug(self, opt, lr, nb_epoch, fc_w, conv_feature_path, conv_feature_file, model_check='val_loss', auto_save=True):
        self.lr = lr
        self.setOpt(opt, self.lr)
        self.ftFC_Block_Aug(nb_epoch=nb_epoch,
                            fc_w=self.model_path+'/'+fc_w,
                            model_check=model_check,
                            conv_feature_path=conv_feature_path,
                            conv_feature_file=conv_feature_file)

    def runEpochs(self, epochs, auto_save=True):
        dn = False
        ISOTIMEFORMAT = '%d_%X'
        while dn is False:
            for epoch in range(epochs):
                print('Running epoch: %d' % (epoch+1))

                self.fit(self.train_batches, self.valid_batches, nb_epoch=1)
                if auto_save is True:
                    curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
                    latest_weights_filename = 'ft%d_%s.h5' % (epoch, curTime)
                    cmd = input_to("To save the weights or not？(Y or N):", 5)
                    if cmd == 'Y':
                        print('Saving the weights in %s.' % latest_weights_filename)
                        self.model.save_weights(self.result_path + '/' + latest_weights_filename)
                    elif cmd == 'N':
                        print('No saving of the weights.')
                    else:
                        if cmd is not None:
                            print('Not valid input. Saving the weights in %s automatically.')
                        else:
                            print('\nSaving the weights in %s automatically.' % latest_weights_filename)
                        self.model.save_weights(self.result_path + '/' + latest_weights_filename)

                        cmd = None

            if input('Enter \'N\' to stop training:') is 'N':
                dn = True
            else:
                num_valid = False
                while num_valid is False:
                    try:
                        y = int(input('num_epochs:'))
                        num_valid = True
                    except:
                        print('Number only!')

                try:
                    lr = float(input('New lr (Previous lr is %d)?' % self.lr))
                    self.lr = lr
                    print('The new learning rate is %s.' % self.lr)
                except:
                    print("Using the previous learning rate.")

    def delMode(self):
        try:
            del self.model
        except:
            print('No model to delete.')

class lrHistory(Callback):
    def on_train_begin(self, logs=None):
        self.lr = []

    def on_epoch_begin(self, epoch, logs=None):
        old_lr = K.get_value(self.model.optimizer.lr)
        if epoch % 5 == 0 and epoch != 0:
            old_lr /= 2
            K.set_value(self.model.optimizer.lr, old_lr)
            print("epoch(%d) 's lr: %f" % ((epoch+1), old_lr))
        self.lr.append(old_lr)
        # print("epoch(%d) 's lr: %f" % (epoch, old_lr))

class saveWeights(Callback):
    def set_lossLine(self, loss):
        self.lossLine = loss

    def getWeightsName(self):
        return self.model_filename

    def on_train_begin(self, logs={}):
        self.losses = []
        self.wsaved = False

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        if self.lossLine is None:
            raise Exception('Set lossLine first.')

        if loss < self.lossLine and self.wsaved is False:
            self.wsaved = True
            self.model_filename = './model_redux/lossLine_%f.hdf5' %loss
            self.model.save_weights(self.model_filename)
            print('Saving weights in file: lossLine_%f.hdf5' %loss)

def train_last_FC_vgg16():
    dc_kaggle = dc_vgg16_bn()
    dc_kaggle.baseModel('vgg16.h5')
    dc_kaggle.trainLastFC(nb_epoch=500, lr=0.00001, opt='Nadam')

def train_FCBlock_vgg16():
    dc_kaggle = dc_vgg16_bn()
    dc_kaggle.baseModel('vgg16.h5')
    dc_kaggle.trainFCBlock(nb_epoch=10, lr=0.000001, opt='Nadam',fc_w='ftFC_Block_08-14_20:52:23.hdf5',
                           model_check='loss')

def train_FCBlock_Aug_vgg16():
    dc_kaggle = dc_vgg16_bn()
    dc_kaggle.baseModel('vgg16.h5')
    dc_kaggle.trainFCBlock_Aug(nb_epoch=20,
                               lr=0.000001,
                               opt='Nadam',
                               fc_w='ftFC_Last_08-13_18:24:26_0.05923.hdf5',
                               # fc_w='a2.hdf5',
                               conv_feature_path='./Trn_data_aug0/',
                               # conv_feature_file='trnConvFeatures_aug4.bc',
                               conv_feature_file='trnConvFeatures.bc',
                               model_check='val_loss')

if __name__ == '__main__':
    train_last_FC_vgg16()
    # train_FCBlock_vgg16()
    # train_FCBlock_Aug_vgg16()
