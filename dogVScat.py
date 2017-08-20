from input_timeout import input_to
from imp import reload
import utils
from keras import backend as K
import vgg16x
import os
import time

class dogs_cats():
    def __init__(self):
        K.set_image_dim_ordering('th')
        reload(vgg16x)

        self.vgg = vgg16x.Vgg16()  # notice here
        self.data_path = os.getcwd()
        print(self.data_path)
        self.train_path = None
        self.valid_path = None
        self.test_path = None
        self.result_path = None
        self.train_batch_size = 32
        self.valid_batch_size = 32
        self.test_batch_size = 32
        self.train_batches = None
        self.valid_batches = None
        self.test_batches = None

        self.no_of_epochs = 1
        self.preds = None

    def setNumEpoch(self, nb):
        self.no_of_epochs = nb

    def setTrainPath(self, str):
        self.train_path = self.data_path + '/' + str

    def setValidPath(self, str):
        self.valid_path = self.data_path + '/' + str

    def setTestPath(self, str):
        self.test_path = self.data_path + '/' + str

    def setResultPath(self, str):
        self.result_path = self.data_path + '/' + str

    def allBatches(self, train_bsize = None, valid_bsize = None, test_bsize = None):
        trSize = (train_bsize if train_bsize is not None else self.train_batch_size)
        vlSize = (valid_bsize if valid_bsize is not None else self.valid_batch_size)
        tsSize = (test_bsize if test_bsize is not None else self.test_batch_size)
        print('Getting train data with the batch size as %s.' % trSize)
        print('Getting valid data with the batch size as %s.' % vlSize)
        print('Getting test data with the batch size as %s.' % tsSize)
        self.train_batches = self.vgg.get_batches(self.train_path, batch_size= trSize)
        self.valid_batches = self.vgg.get_batches(self.valid_path, batch_size= vlSize)
        self.test_batches = self.vgg.get_batches(self.test_path, shuffle=False, batch_size= tsSize, class_mode=None)

    def fineTune(self):
        if self.train_batches is None:
            print('The VGG model cannot finetune because that the train batches are none.')
        else:
            self.vgg.finetune(self.train_batches)

    def fit(self, auto_save_w=True):
        ISOTIMEFORMAT = '%y-%m-%d_%X'
        latest_weights_filename = None
        epochs_done = 0
        cmd = None

        for epoch in range(self.no_of_epochs):
            epochs_done = epoch + 1
            print('Running epoch: %d' % epochs_done)

            # fit
            self.vgg.fit(self.train_batches, self.valid_batches, nb_epoch=1)
            # to save weights
            if auto_save_w is True:
                curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
                latest_weights_filename = 'ft%d_%s.h5' % (epoch, curTime)
                cmd = input_to("To save the weights or not？(Y or N):", 4)
                if cmd == 'Y':
                    print('Saving the weights in %s.' % latest_weights_filename)
                    self.vgg.model.save_weights(self.result_path + '/' +latest_weights_filename)
                elif cmd == 'N':
                    print('No saving of the weights.')
                else:
                    if cmd is not None:
                        print('Not valid input. Saving the weights in %s automatically.')
                    else:
                        print('\nSaving the weights in %s automatically.' % latest_weights_filename)
                    self.vgg.model.save_weights(self.result_path + '/' +latest_weights_filename)

                cmd = None
            # check if needs to stop
            if epoch < (self.no_of_epochs - 1):
                cmd = input_to("Input 'N' to stop fitting:", 3)
                if cmd == 'N':
                    break
                else:
                    if cmd is not None:
                        print('\nNot valid input. Continue fitting...')
                    else:
                        print('\nContinue fitting...')
                cmd = None

        print('Completed %s fit operations' % epochs_done)

    def test(self, save=False):
        self.preds = self.vgg.test(self.test_batches)
        if save:
            ISOTIMEFORMAT = '%y-%m-%d_%X'
            curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
            utils.save_array(self.result_path + '/' + 'test_preds_%s' %curTime, self.preds)
            utils.save_array(self.result_path + '/' + 'filenames_%s' %curTime, self.test_batches.filenames)

    def results(self, num):
        print(self.preds[:num])
        print(self.test_batches.filenames[:num])

    def loadWeights(self, weightsFile):
        self.vgg.model.load_weights(self.result_path+'/'+weightsFile)

def op_0():
    dnc = dogs_cats()
    dnc.setTrainPath('train_redux')
    dnc.setValidPath('valid_redux')
    dnc.setTestPath('test_redux')
    dnc.setResultPath('results_redux')
    dnc.allBatches(train_bsize=128, valid_bsize=128, test_bsize=100)
    dnc.fineTune()
    dnc.setNumEpoch(3)
    dnc.fit()
    dnc.test(save=True)
    dnc.results(5)

def op_1():
    dnc = dogs_cats()
    dnc.setTrainPath('train_redux')
    dnc.setValidPath('valid_redux')
    dnc.setTestPath('test_redux')
    dnc.setResultPath('results_redux')
    dnc.allBatches(train_bsize=128, valid_bsize=128, test_bsize=100)
    dnc.fineTune()
    dnc.loadWeights('ft7_17-08-02_22:50:23.h5')
    dnc.test(save=True)

if __name__ == '__main__':
    op_0()
