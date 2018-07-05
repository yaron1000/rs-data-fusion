import keras
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Dense,
    Activation,
    Lambda
)
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np

##################################################################
# Deep Convolutional SpatioTemporal Fusion Network (DCSTFN)
##################################################################


def dcstfn(coarse_input, fine_input, coarse_pred, d=[32, 64, 128]):
    pool_size = 2
    assert coarse_input.shape == coarse_pred.shape
    coarse_model = _htls_cnet(coarse_input, d)
    fine_model = _hslt_cnet(fine_input, d)

    # 三个网络的融合
    coarse_input_layer = Input(shape=coarse_input.shape[-3:])
    coarse_input_model = coarse_model(coarse_input_layer)
    fine_input_layer = Input(shape=fine_input.shape[-3:])
    fine_input_model = fine_model(fine_input_layer)
    subtracted_layer = keras.layers.subtract([fine_input_model, coarse_input_model])
    coarse_pred_layer = Input(shape=coarse_pred.shape[-3:])
    coarse_pred_model = coarse_model(coarse_pred_layer)
    added_layer = keras.layers.add([subtracted_layer, coarse_pred_model])
    merged_layer = Conv2DTranspose(d[1], 3, strides=pool_size,
                                   padding='same',
                                   kernel_initializer='he_normal',
                                   activation='relu')(added_layer)
    dense_layer = Dense(d[0], activation='relu')(merged_layer)
    final_out = Dense(fine_input.shape[-1])(dense_layer)
    model = Model([coarse_input_layer, fine_input_layer, coarse_pred_layer], final_out)
    return model


def _hslt_cnet(fine_input, d, pool_size=2):
    # 对于Landsat高分辨率影像建立网络
    fine_model = Sequential()
    fine_model.add(Conv2D(d[0], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu', input_shape=fine_input.shape[-3:]))
    fine_model.add(Conv2D(d[1], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    fine_model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    fine_model.add(Conv2D(d[1], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    fine_model.add(Conv2D(d[2], 3, padding='same',
                          kernel_initializer='he_normal',
                          activation='relu'))
    return fine_model


def _htls_cnet(coarse_input, d):
    # 对于MODIS影像建立相同的网络
    coarse_model = Sequential()
    coarse_model.add(Conv2D(d[0], 3, padding='same',
                            kernel_initializer='he_normal',
                            activation='relu', input_shape=coarse_input.shape[-3:]))
    coarse_model.add(Conv2D(d[1], 3, padding='same',
                            kernel_initializer='he_normal',
                            activation='relu'))
    for n in [2, 2, 2]:
        coarse_model.add(Conv2DTranspose(d[1], 3, strides=n, padding='same',
                                         kernel_initializer='he_normal'))
    coarse_model.add(Conv2D(d[2], 3, padding='same',
                            kernel_initializer='he_normal',
                            activation='relu'))
    return coarse_model


##################################################################
# Deep Residual SpatioTemporal Fusion Network (DRSTFN)
##################################################################

def _conv_conv_block(input_tensor, n_features):
    assert len(n_features) == 2
    norm = BatchNormalization()(input_tensor)
    relu = Activation('relu')(norm)
    conv1 = Conv2D(n_features[0], 3, padding='same',
                   kernel_initializer='he_normal')(relu)
    norm = BatchNormalization()(conv1)
    relu = Activation('relu')(norm)
    conv2 = Conv2D(n_features[1], 3, padding='same',
                   kernel_initializer='he_normal')(relu)
    return conv2


def _conv_deconv_block(input_tensor, n_features):
    assert len(n_features) == 2
    norm = BatchNormalization()(input_tensor)
    relu = Activation('relu')(norm)
    conv1 = Conv2D(n_features[0], 3, padding='same',
                   kernel_initializer='he_normal')(relu)
    norm = BatchNormalization()(conv1)
    relu = Activation('relu')(norm)
    conv2 = Conv2DTranspose(n_features[1], 3, strides=2, padding='same',
                            kernel_initializer='he_normal')(relu)
    return conv2


def _shortcut(input_tensor, residual):
    shortcut = input_tensor
    input_shape = K.int_shape(input_tensor)
    residual_shape = K.int_shape(residual)

    # 维度不一样
    if input_shape[-1] != residual_shape[-1]:
        shortcut = Conv2D(residual_shape[-1], 1, padding='same',
                          kernel_initializer='he_normal')(shortcut)

    # 尺寸不一样
    if input_shape[-2] != residual_shape[-2] or input_shape[-3] != residual_shape[-3]:
        shortcut = Lambda(lambda im: K.tf.image.resize_images(im,
                                                              (residual_shape[-2], residual_shape[-3]),
                                                              method=K.tf.image.ResizeMethod.BICUBIC
                                                              ))(shortcut)

    return keras.layers.add([shortcut, residual])


def _resnet(input_tensor, d, res_block):
    input_layer = Input(shape=input_tensor.shape[-3:])
    model = Conv2D(d[0], 3, padding='same',
                   kernel_initializer='he_normal')(input_layer)
    model = BatchNormalization()(model)
    model = Activation('relu')(model)
    for i in range(1, len(d), 2):
        residual = res_block(model, (d[i], d[i + 1]))
        model = _shortcut(model, residual)
    model = Model(input_layer, model)
    return model


def _hslt_rnet(fine_input, d):
    return _resnet(fine_input, d, _conv_conv_block)


def _htls_rnet(coarse_input, d):
    return _resnet(coarse_input, d, _conv_deconv_block)


def drstfn(coarse_input, fine_input, coarse_pred, d=[32, 64, 128]):
    assert coarse_input.shape == coarse_pred.shape
    dd = [d[0]] + list(np.repeat(d[1:], 4))

    coarse_model = _htls_rnet(coarse_input, dd)
    fine_model = _hslt_rnet(fine_input, dd)

    # 三个网络的融合
    coarse_input_layer = Input(shape=coarse_input.shape[-3:])
    coarse_input_model = coarse_model(coarse_input_layer)
    fine_input_layer = Input(shape=fine_input.shape[-3:])
    fine_input_model = fine_model(fine_input_layer)
    subtracted_layer = keras.layers.subtract([fine_input_model, coarse_input_model])

    coarse_pred_layer = Input(shape=coarse_pred.shape[-3:])
    coarse_pred_model = coarse_model(coarse_pred_layer)
    added_layer = keras.layers.add([subtracted_layer, coarse_pred_model])

    merged_layer = added_layer
    dd = list(np.repeat(d[::-1][1:], 2))
    for i in range(0, len(dd), 2):
        residual = _conv_conv_block(merged_layer, (dd[i], dd[i + 1]))
        merged_layer = _shortcut(merged_layer, residual)
    final_out = Dense(fine_input.shape[-1])(merged_layer)
    model = Model([coarse_input_layer, fine_input_layer, coarse_pred_layer], final_out)
    return model



def get_model(name):
    """通过字符串形式的函数名称得到该函数对象，可以直接对该函数进行调用"""
    return globals()[name]
