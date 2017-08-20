import os, subprocess, glob
import numpy as np
from shutil import copyfile, rmtree
import shutil

class DataBuilder():
    def __init__(self):
        self.DATA_HOME_DIR = os.getcwd()
        self.valid_size = None
        self.sample_train_size = None
        self.sample_valid_size = None

    def setValidSize(self, size):
        self.valid_size = size
        print('valid_size is set to %d.' % self.valid_size)

    def setSampleValidSize(self, size):
        self.sample_valid_size = size
        print('sample_valid_size is set to %d.' % self.sample_valid_size)

    def setSampleTrainSize(self, size):
        self.sample_train_size = size
        print('sample_train_size is set to %d.' % self.sample_train_size)

    def createDirs(self):
        subprocess.call(["mkdir", "valid_redux"])
        subprocess.call(["mkdir", "results_redux"])
        subprocess.call(["mkdir", "-p", "test_redux/unknown"])
        subprocess.call(["mkdir", "-p", "sample/train"])
        subprocess.call(["mkdir", "-p", "sample/test"])
        subprocess.call(["mkdir", "-p", "sample/valid"])
        subprocess.call(["mkdir", "-p", "sample/results"])

        self.dir_train = 'train_redux'
        self.dir_train_cats = 'train_redux/cats'
        self.dir_train_dogs = 'train_redux/dogs'
        self.dir_valid = 'valid_redux'
        self.dir_valid_cats = 'valid_redux/cats'
        self.dir_valid_dogs = 'valid_redux/dogs'
        self.dir_test = 'test_redux'
        self.dir_test_unknown = 'test_redux/unknown'
        self.dir_results = 'results_redux'

        self.dir_sample_train = 'sample/train'
        self.dir_sample_train_cats = "sample/train/cats"
        self.dir_sample_train_dogs = "sample/train/dogs"
        self.dir_sample_valid = 'sample/valid'
        self.dir_sample_valid_cats = "sample/valid/cats"
        self.dir_sample_valid_dogs = "sample/valid/dogs"
        self.dir_sample_test = 'sample/test'
        self.dir_sample_results = 'sample/results'

    def build(self):
        if self.valid_size is None:
            print("Build fails because that the valid size is not set yet.")
            print("Using method 'setValidSize' to set it.")
            return

        if self.sample_train_size is None:
            print("Build fails because that the sample train size is not set yet.")
            print("Using method 'setSampleTrainSize' to set it.")
            return

        if self.sample_valid_size is None:
            print("Build fails because that the sample valid size is not set yet.")
            print("Using method 'setSampleValidSize' to set it.")
            return

        print('Creating all the directories...')
        self.createDirs()
        print('Importing data into the valid directory...')
        self.importData()
        print('Building sample data sets...')
        self.buildSampleData()
        print('Rearrange all the data into classess...')
        self.reArrange()
        print('Building done.')

    def buildSampleData(self):
        print("Import data to the SAMPLE directories as SAMPLES...")
        # codes here
        self.dataCopy(dir_src=self.dir_train, dir_dst=self.dir_sample_train, size=self.sample_train_size)
        self.dataCopy(dir_src=self.dir_valid, dir_dst=self.dir_sample_valid, size=self.sample_valid_size)

        print("Importing data is done.")

    def importData(self):
        print("Import data to the related directories...")
        # codes here
        self.dataTransfer(dst_num=self.valid_size, dir_dst=self.dir_valid, dir_src=self.dir_train, file_type='*.jpg')
        print("Importing data is done.")

    def reArrange(self):
        # for real
        subprocess.call(["mkdir", "-p", self.dir_train_cats])
        subprocess.call(["mkdir", "-p", self.dir_train_dogs])
        self.dataTransfer(dir_dst=self.dir_train_cats, dir_src=self.dir_train, file_type='cat.*.jpg')
        self.dataTransfer(dir_dst=self.dir_train_dogs, dir_src=self.dir_train, file_type='dog.*.jpg')

        subprocess.call(["mkdir", "-p", self.dir_valid_cats])
        subprocess.call(["mkdir", "-p", self.dir_valid_dogs])
        self.dataTransfer(dir_dst=self.dir_valid_cats, dir_src=self.dir_valid, file_type='cat.*.jpg')
        self.dataTransfer(dir_dst=self.dir_valid_dogs, dir_src=self.dir_valid, file_type='dog.*.jpg')

        subprocess.call(["mkdir", "-p", self.dir_test_unknown])
        self.dataTransfer(dir_dst=self.dir_test_unknown, dir_src=self.dir_test, file_type='*.jpg')

        # for the samples
        subprocess.call(["mkdir", "-p", self.dir_sample_train_cats])
        subprocess.call(["mkdir", "-p", self.dir_sample_train_dogs])
        self.dataTransfer(dir_dst=self.dir_sample_train_cats, dir_src=self.dir_sample_train, file_type='cat.*.jpg')
        self.dataTransfer(dir_dst=self.dir_sample_train_dogs, dir_src=self.dir_sample_train, file_type='dog.*.jpg')

        subprocess.call(["mkdir", "-p", self.dir_sample_valid_cats])
        subprocess.call(["mkdir", "-p", self.dir_sample_valid_dogs])
        self.dataTransfer(dir_dst=self.dir_sample_valid_cats, dir_src=self.dir_sample_valid, file_type='cat.*.jpg')
        self.dataTransfer(dir_dst=self.dir_sample_valid_dogs, dir_src=self.dir_sample_valid, file_type='dog.*.jpg')

    def dataTransfer(self, dir_src, dir_dst, file_type, dst_num=None):
        s = glob.glob(self.DATA_HOME_DIR+'/'+dir_src+'/'+file_type)
        d = glob.glob(self.DATA_HOME_DIR+'/'+dir_dst+'/'+file_type)

        nb_s = len(s)
        nb_d = len(d)
        print("%d '%s' items are found." %(nb_s, dir_src))
        print("%d '%s' items are found." %(nb_d, dir_dst))

        if dst_num is None:
            num_trans = nb_s
        else:
            if dst_num <= 0:
                raise Exception("Valid size must be positive number.")
            num_trans = dst_num - nb_d

        if num_trans == 0:
            print("The number of '%s' items meets requirement." % dir_dst)
        elif num_trans < 0:
            print("%d items are moved from '%s' to '%s'." %(-num_trans, dir_dst, dir_src))
            shuf_v = np.random.permutation(d)
            for i in range(-num_trans):
                os.rename(shuf_v[i], self.DATA_HOME_DIR+'/'+dir_src+'/'+os.path.split(shuf_v[i])[-1])
        elif num_trans >0 and num_trans <= nb_s:
            print("%d items are moved from '%s' to '%s'." %(num_trans, dir_src, dir_dst))
            shuf_t = np.random.permutation(s)
            for i in range(num_trans):
                os.rename(shuf_t[i], self.DATA_HOME_DIR+'/'+dir_dst+'/'+os.path.split(shuf_t[i])[-1])
        else:
            raise Exception("Not enough items to move from %s to %s." %(dir_src, dir_dst))

    def dataCopy(self, dir_src, dir_dst, size):
        subprocess.call("rm -r " + dir_dst + "/*.jpg", shell=True)

        g = glob.glob(self.DATA_HOME_DIR + '/' + dir_src + '/' + '*.jpg')
        nb_t = len(g)

        if size > nb_t:
            raise Exception('The size of copying set exceeds the size of source set.')

        shuf = np.random.permutation(g)
        for i in range(size):
            copyfile(shuf[i], self.DATA_HOME_DIR + '/'+ dir_dst + '/' + os.path.split(shuf[i])[-1])


def main():
    DB_dogscats = DataBuilder()
    DB_dogscats.setValidSize(2000)
    DB_dogscats.setSampleTrainSize(2300)
    DB_dogscats.setSampleValidSize(200)
    DB_dogscats.build()

if __name__ == '__main__':
    main()
