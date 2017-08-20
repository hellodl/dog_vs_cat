from keras import backend as K
K.set_image_dim_ordering('th')

from keras.models import Sequential
from keras.layers.core import Lambda, Flatten, Dense, Dropout
from keras.layers.convolutional import ZeroPadding2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam, Nadam, Adadelta, Adamax, Adagrad

import numpy as np


def check_path(path):
    if path is None:
        raise Exception('\'path\' is not configured.')
    pathList = ['Train', 'Valid', 'Test', 'Result', 'ModelOut','Sub']
    for item in pathList:
        if path.get(item, None) is None:
            raise Exception('The required path \'%s\' is not configured.' % item)

    print('All the paths are found.')

def check_batch_size(batchSize):
    if batchSize is None:
        raise Exception('\'batch size\' is not configured.')
    sizeList = ['Train', 'Valid', 'Test']
    for item in sizeList:
        size = batchSize.get(item, None)
        if size is None:
            raise Exception('The required batch size \'%s\' is not configured' % item)
        if type(size) is not int:
            raise Exception('The configuration of %s size must be a \'int\' type.' % item)

def check_hyper_parameter(hparams):
    if hparams is None:
        raise Exception('\'hyper_parameter\' is not configured.')
    hparamsList = ['Optimizer']
    for item in hparamsList:
        if hparams.get(item, None) is None:
            raise Exception('The required \'%s\' is not configured.' % item)


def check_cfg(model_cfg):
    check_path(model_cfg.get('path', None))
    check_batch_size(model_cfg.get('batch_size', None))
    check_hyper_parameter(model_cfg.get('hyper_parameter', None))


def args_default(args_list, optimizer):
    if 'lr' in args_list:
        print(' >> lr = %02.1e' % K.get_value(optimizer.lr))

    if 'momentum' in args_list:
        print(' >> momentum = %03.2f' % K.get_value(optimizer.momentum))

    if 'decay' in args_list:
        print(" >> decay = %03.2f" % K.get_value(optimizer.decay))

    if 'nesterov' in args_list:
        print(" >> nesterov = %s" % optimizer.nesterov)

    if 'rho' in args_list:
        print(' >> rho = %3.2f' % K.get_value(optimizer.rho))

    if 'epsilon' in args_list:
        print(" >> epsilon = %02.1e" % optimizer.epsilon)

    if 'beta_1' in args_list:
        print(' >> beta_1 = %03.2f' % K.get_value(optimizer.beta_1))

    if 'beta_2' in args_list:
        print(" >> beta_2 = %05.4f" % K.get_value(optimizer.beta_2))

    if 'schedule_decay' in args_list:
        print(" >> schedule_decay = %05.4e" % optimizer.schedule_decay)

def args_setup(args_list, args, optimizer):
    print('From configuration:')
    if 'lr' in args_list:
        lr = args.pop('lr', None)
        if lr is not None:
            K.set_value(optimizer.lr, lr)
            print(' >> lr is set to %02.1e.' % K.get_value(optimizer.lr))

    if 'momentum' in args_list:
        momentum = args.pop('momentum', None)
        if momentum is not None:
            K.set_value(optimizer.momentum, momentum)
            print(' >> momentum is set to %03.2f.' % K.get_value(optimizer.momentum))

    if 'decay' in args_list:
        decay = args.pop('decay', None)
        if decay is not None:
            K.set_value(optimizer.decay, decay)
            print(' >> decay is set to %03.2f.' % K.get_value(optimizer.decay))

    if 'nesterov' in args_list:
        nesterov = args.pop('nesterov', None)
        if nesterov is not None:
            optimizer.nesterov = nesterov
            print(" >> nesterov = %s" % optimizer.nesterov)

    if 'rho' in args_list:
        rho = args.pop('rho',None)
        if rho is not None:
            K.set_value(optimizer.rho, rho)
            print(' >> rho = %3.2f' % K.get_value(optimizer.rho))

    if 'epsilon' in args_list:
        epsilon = args.pop('epsilon', None)
        if epsilon is not None:
            optimizer.epsilon = epsilon
            print(" >> epsilon = %02.1e" % optimizer.epsilon)

    if 'beta_1' in args_list:
        beta_1 = args.pop('beta1', None)
        if beta_1 is not None:
            K.set_value(optimizer.beta_1, beta_1)
            print(' >> beta_1 = %03.2f' % K.get_value(optimizer.beta_1))

    if 'beta_2' in args_list:
        beta_2 = args.pop('beta2', None)
        if beta_2 is not None:
            K.set_value(optimizer.beta_2, beta_2)
            print(" >> beta_2 = %05.4f" % K.get_value(optimizer.beta_2))

    if 'schedule_decay' in args_list:
        schedule_decay = args.pop('schedule_decay', None)
        if schedule_decay is not None:
            optimizer.schedule_decay = schedule_decay
            print(" >> schedule_decay = %05.4e" % optimizer.schedule_decay)

    for k, v in args.items():
        print(" >> Argument '%s' is ignored." % k)


def optimizer_cfg(model_cfg):
    opt_params = {'SGD':{'class':SGD(),
                         'arg_list':['lr', 'momentum', 'decay', 'nesterov']
                        },
                  'RMSprop': {'class': RMSprop(),
                              'arg_list': ['lr', 'rho', 'epsilon', 'decay']
                             },
                  'Adagrad': {'class': Adagrad(),
                              'arg_list': ['lr','epsilon','decay']
                             },
                  'Adadelta': {'class': Adadelta(),
                              'arg_list': ['lr', 'rho', 'epsilon', 'decay']
                              },
                  'Adam': {'class': Adam(),
                           'arg_list': ['lr', 'rho', 'epsilon', 'decay']
                          },
                  'Adamax': {'class': Adamax(),
                             'arg_list': ['lr', 'beta_1', 'beta_2', 'epsilon', 'decay']
                            },
                  'Nadam': {'class': Nadam(),
                            'arg_list': ['lr', 'beta_1', 'beta_2', 'epsilon', 'schedule_decay']
                           }
                 }

    opt = model_cfg.get('hyper_parameter').get('Optimizer')
    if opt.get('type', None) is None:
        raise Exception('The type of optimizer is not configured.')

    optimizer = None
    print('The optimizer selected is %s.' % opt['type'])
    print('The default arguments are:')
    if opt_params.get(opt['type'], None) is not None:
        optimizer = opt_params[opt['type']]['class']
        args_default(opt_params[opt['type']]['arg_list'], optimizer)
        args_cfg = opt.get('args', None)
        if args_cfg is not None:
            args_setup(opt_params[opt['type']]['arg_list'], args_cfg, optimizer)
    else:
        raise Exception('The optimizer \'%s\' is not supported.' % opt['type'])

    return optimizer


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


def ConvBlock(model, nb_block, nb_layer, nb_filter, activation):
    for i in range(nb_block):
        for j in range(nb_layer[i]):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(nb_filter[i], (3, 3), activation=activation))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCBlock(model, classes, nb_neuron, dropout, activation):
    model.add(Flatten())
    for i in range(len(nb_neuron)):
        model.add(Dense(nb_neuron[i], activation=activation[i]))
        model.add(Dropout(dropout[i]))

    model.add(Dense(classes, activation='softmax'))

def build_vgg16(weights):
    model = Sequential()
    model.add(
        Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    ConvBlock(model,
               nb_block = 5,
               nb_layer = [2, 2, 3, 3, 3],
               nb_filter=[64, 128, 256, 512, 512],
               activation='relu')

    FCBlock(model,
            classes=1000,
            nb_neuron=[4096, 4096],
            dropout=[0.5, 0.5],
            activation=['relu', 'relu', 'softmax'])


    model.load_weights(weights)  # 'vgg16.h5'
    model.pop()
    return model


def build_top(model, top, classes):
    if top == 'default':
        model.add(Dense(classes, activation='softmax'))
    else:
        neuron_l = top.get('neurons')
        dropout_l = top.get('dropout')
        activation_l = top.get('activation')
        FCBlock(model,
                classes=2,
                nb_neuron=neuron_l,
                dropout=dropout_l,
                activation=activation_l)


def set_trainable(model, finetune):
    if finetune == 'default':
        return

    layers={'Conv2':5,
            'Conv3':10,
            'Conv4':17,
            'Conv5':24,
            'FC_block':31,
            'FC_last':len(model.layers)-2
    }
    layer_end = layers[finetune]
    # print(model.layers[layer_end])

    for layer in model.layers[: layer_end]:
        layer.trainable = False

def build_model(settings):
    model = None
    model_setup = settings.get('model')
    base_w = model_setup.get('base')
    top = model_setup.get('top')
    classes = model_setup.get('classes')
    finetune = model_setup.get('finetune')

    # build basic vgg16
    base_type = base_w[:base_w.find('.')]
    if base_type == 'vgg16':
        print('Using \'vgg16\' as base model.')
        model = build_vgg16(base_w)

    else:
        raise Exception('Other base is not supported.')

    # modify the fc_block
    build_top(model, top, classes)
    set_trainable(model, finetune)

    return model
