import bcolz
import utils
import os
import numpy as np
import time

cur_dir = os.getcwd()
results_path = cur_dir + '/results_redux/'
print(results_path)

preds = utils.load_array(results_path + 'test_preds_17-08-02_23:59:56')
filenames = utils.load_array(results_path + 'filenames_17-08-02_23:59:56')

isdog = preds[:,1]
isdog = isdog.clip(min=0.05, max=0.95)
ids = np.array([int(f[8:f.find('.')]) for f in filenames])
subm = np.stack([ids,isdog], axis=1)

ISOTIMEFORMAT = '%y-%m-%d_%X'
curTime = time.strftime(ISOTIMEFORMAT, time.localtime())
submission_file_name = 'submission_%s.csv' %curTime
np.savetxt(submission_file_name, subm, fmt='%d,%.5f', header='id,label', comments='')
