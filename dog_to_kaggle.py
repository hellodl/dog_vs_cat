import numpy as np
from dog_utils import check_cfg, optimizer_cfg, build_model

class dog_to_kaggle():
    def __init__(self, model_cfg):
        check_cfg(model_cfg)
        self.settings = model_cfg
        self.opt = optimizer_cfg(self.settings)
        self.model = build_model(self.settings)
        # print(self.model.summary())
        # print(self.model.layers[2])
        # print(self.model.layers[3])

    def train_model(self):
        pass

    def predict(self):
        pass

def ftLastFC():
    print("Finetune the last FC of VGG16 to predict 'CAT' or 'DOG'.")
    conf = {
            'path':{
                'Train':'./train_redux',
                'Valid':'./valid_redux',
                'Test':'./test_redux',
                'Result':'./reuslt_redux',
                'ModelOut':'./model_redux',
                'Sub':'./sub_redux'
            },
            'batch_size':{
                'Train': 64,
                'Valid': 64,
                'Test': 64
            },
            'hyper_parameter':{
                'Optimizer':{
                             'type':'Nadam',
                             'args':{
                                     'lr':0.00001,
                             }
                }
            },
            'model':{
                     'base':'vgg16.h5',
                     'top':'default',
                    # 'top': {'neurons':[4096, 4096],
                    #         'dropout':[0.5, 0.5],
                    #         'activation':['relu', 'relu']},
                     'classes':2,
                     'finetune':'FC_last'  # or 'FC_block' 'Conv5'...'Conv2' 'default'
            },
            'train':{
                     'epoch':20,
                     'bottleneck':True,
                     'bottleneck_files':{
                         'train_model':'xx.hdf5',
                         'trn_features':'trnConvFeatures.bc',
                         'val_features':'valConvFeatures.bc',
                         'tst_features':'tstConvFeatures.bc',
                         'trn_labels':'trnLabels.bc',
                         'val_labels':'valLabels.bc',
                         'test_id':'test_id.bc'
                     },
                     'data_aug':True,
                     'aug_args':{
                                 'rotation_range':20,
                                 'shear_range':0.1,
                                 'zoom_range':[0.8, 1.05],
                                 'width_shift_range':0.1,
                                 'height_shift_range':0.1,
                                 'horizontal_flip':True,
                                 'fill_mode':'constant'
                     },
                     'callback':{
                         'ModelCheckpoint':{
                             'filepath':'path',
                             'monitor':'val_acc',  # 'loss' 'acc' or 'val_loss'
                             'verbose':1,
                             'save_best_only':True,
                         },
                         'EarlyStopping':None,
                         'ReduceLROnPlateau':None,
                         'LearningRateScheduler':None,
                         'CSVLogger':None,
                         'customize':['xk']
                     },
            },
            'time_format':'%m-%d_%X'
    }

    dog0 = dog_to_kaggle(model_cfg=conf)

if __name__ == "__main__":
    ftLastFC()
