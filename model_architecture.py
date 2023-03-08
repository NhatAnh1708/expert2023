#!/usr/bin/python3
'''ShuffleNet Architecture constructor.

This module implements function, which constructs Convolutional neural network for
metric learning based on ShuffleNetV2 (https://arxiv.org/abs/1807.11164).

This code uses the implementation in Keras from repository:
https://github.com/opconty/keras-shufflenetV2

Code have minor changes, which allows to interpolate the bottleneck ratio.

Examples:
    To use this module, you simply import class in your python code:
        # from model_architecture import build_network

    To build a model for images with sizes 64x64x3, use the following code:
        # model = build_network(input_shape=(64, 64, 3), embedding_size=16)

Todo:
    * Add more functionality

.. _Expert eyeglasses recommendation system with Generative Adversarial Networks:
   https://github.com/Defasium/expertglasses

'''

from keras import backend as K
from keras.models import Model
from keras.layers import Lambda, Dense, Dropout
import numpy as np
from keras.utils import plot_model
from keras.applications.imagenet_utils import obtain_input_shape
from keras.utils import get_source_inputs
from keras.layers import Input, Conv2D, MaxPool2D, GlobalMaxPooling2D, GlobalAveragePooling2D,BatchNormalization,DepthwiseConv2D,Concatenate
from keras.layers import Activation, Dense
from keras.models import Model

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x
def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1, act="relu", pre="", batchnorm=True):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = '{}stage{}/block{}'.format(pre, stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation(act, name='{}/{}_1x1conv_1'.format(prefix, act))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    if batchnorm:
        x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation(act, name='{}/{}_1x1conv_2'.format(prefix, act))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        if batchnorm:
            s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        if batchnorm:
            s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation(act, name='{}/{}_1x1conv_3'.format(prefix, act))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret
def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1, act="relu", first_stride=2, prefix="", batchnorm=True):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                      strides=first_stride,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1, act=act, pre=prefix, batchnorm=batchnorm)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], strides=1,
                          bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i), act=act, pre=prefix, batchnorm=batchnorm)

    return x
def ShuffleNetV2(include_top=True,
                 input_tensor=None,
                 scale_factor=1.0,
                 pooling='max',
                 input_shape=(224, 224, 3),
                 load_model=None,
                 num_shuffle_units=[3, 7, 3],
                 bottleneck_ratio=1,
                 classes=1000,
                 activation="relu"):
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only tensorflow supported for now')
    name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
    input_shape = obtain_input_shape(input_shape, default_size=224, min_size=28, require_flatten=include_top,
                                      data_format=K.image_data_format())
    out_dim_stage_two = {0.5: 48, 1: 116, 1.5: 176, 2: 244}

    if pooling not in ['max', 'avg']:
        raise ValueError('Invalid value for pooling')
    if not (float(scale_factor) * 4).is_integer():
        raise ValueError('Invalid value for scale_factor, should be x over 4')
    exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
    out_channels_in_stage = 2 ** exp
    try:
        out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage
    except KeyError:
        out_channels_in_stage *= int((out_dim_stage_two[2] - out_dim_stage_two[0.5]
                                      ) / 1.5 * (bottleneck_ratio - 0.5)) + out_dim_stage_two[0.5]  # interpolate
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # create shufflenet architecture
    x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
               activation='relu', name='conv1')(img_input)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage, repeat in enumerate(num_shuffle_units):
        x = block(x, out_channels_in_stage,
                  repeat=repeat,
                  bottleneck_ratio=bottleneck_ratio,
                  stage=stage + 2)

    if bottleneck_ratio < 1:
        k = 512
    elif bottleneck_ratio < 2:
        k = 1024
    else:
        k = 2048
    x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='global_max_pool')(x)

    if include_top:
        x = Dense(classes, name='fc')(x)
        x = Activation('softmax', name='softmax')(x)
    else:
        return img_input, x
    if input_tensor:
        inputs = get_source_inputs(input_tensor)

    else:
        inputs = img_input

    model = Model(inputs, x, name=name)

    if load_model:
        model.load_weights('', by_name=True)

    return model


def build_network(input_shape, embedding_size):
    '''Api-request to face++ to get various attributes and head orientation.

                Args:
                    input_shape (tuple of int): Input shape of images.
                    embedding_size (int): Size of the final embedding layer.

                Returns:
                    model (tensorflow.keras.engine.training.Model): Keras model.

    '''
    inputs, outputs = ShuffleNetV2(include_top=False, input_shape=input_shape,
                                   bottleneck_ratio=0.35, num_shuffle_units=[2, 2, 2])
    outputs = Dropout(0.0)(outputs)
    outputs = Dense(embedding_size, activation=None,
                    kernel_initializer='he_uniform')(outputs)
    # force the encoding to live on the d-dimentional hypershpere
    outputs = Lambda(lambda x: K.l2_normalize(x, axis=-1))(outputs)
    return Model(inputs, outputs)
