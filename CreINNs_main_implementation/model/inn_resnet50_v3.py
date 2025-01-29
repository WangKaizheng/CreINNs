from model.inn_layers_v3 import IntDense, IntConv2D, IntFlatten, IntBatchNormalization, IntAdd
from model.inn_layers_v3 import IntDropout, IntAveragePooling2D, IntMaxPooling2D, IntGlobalAveragePooling2D, IntZeroPadding2D
from model.inn_layers_v3 import IntSoftMax, IntRelu, IntSoftMaxSingle
from keras.models import Model
from keras.layers import Activation
from keras import Input
import tensorflow as tf
from tensorflow import keras
import numpy as np


def _identity_block(inputs, channels):

    INIT_SCHEME ='glorot_uniform'
    # INIT_SCHEME_2 = 'glorot_normal'
    INIT_SCHEME_2 = 'glorot_uniform'

    x = IntConv2D(channels, strides=(1, 1), kernel_size=(1, 1), padding="valid",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(inputs)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)
    x = IntRelu()(x)

    x = IntConv2D(channels, strides=(1, 1), kernel_size=(3, 3), padding="same",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(x)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)
    x = IntRelu()(x)

    x = IntConv2D(4*channels, strides=(1, 1), kernel_size=(1, 1), padding="valid",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(x)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)

    x = IntAdd()([inputs, x])
    out = IntRelu()(x)

    return out


def _conv_block(inputs, channels, strides=(2, 2)):

    INIT_SCHEME ='glorot_uniform'
    # INIT_SCHEME_2 = 'glorot_normal'
    INIT_SCHEME_2 = 'glorot_uniform'
    
    x = IntConv2D(channels, strides=strides, kernel_size=(1, 1), padding="valid",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(inputs)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)
    x = IntRelu()(x)

    x = IntConv2D(channels, strides=(1, 1), kernel_size=(3, 3), padding="same",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(x)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)
    x = IntRelu()(x)

    x = IntConv2D(4*channels, strides=(1, 1), kernel_size=(1, 1), padding="valid",
                    center_kernel_initializer=INIT_SCHEME, radius_kernel_initializer=INIT_SCHEME_2)(x)
    x = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(x)

    # Downsampling
    inputs_down = IntConv2D(4*channels, strides=strides, kernel_size=(1, 1), center_kernel_initializer=INIT_SCHEME,
        radius_kernel_initializer=INIT_SCHEME_2, padding="valid")(inputs)
    inputs_down = IntBatchNormalization(momentum = 0.9, epsilon=1e-5)(inputs_down)

    x = IntAdd()([inputs_down, x])
    
    out = IntRelu()(x)
    
    return out
    
def inn_resnet50(input_shape, num_classes, predict_mod):
    
    inputs = [Input(input_shape), Input(input_shape)]
    
    # Initial layers
    out = IntZeroPadding2D(padding=3)(inputs)
    out = IntConv2D(64, (7, 7), strides=(2, 2),padding="valid", center_kernel_initializer="he_normal", radius_kernel_initializer="zeros")(out)
    out = IntBatchNormalization()(out)
    out = IntRelu()(out)
    out = IntZeroPadding2D(padding=1)(out)
    out = IntMaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(out)

    # Resnet blocks
    chanels_per_block = [64, 128, 256, 512]
    block_repeat = [3, 4, 6, 3]
    for k in range(len(chanels_per_block)):
        for i in range(block_repeat[k]):
            if i == 0:
                if k == 0:
                   out = _conv_block(out, channels=chanels_per_block[k], strides=(1, 1))
                else:
                   out = _conv_block(out, channels=chanels_per_block[k], strides=(2, 2))
            else:
                out = _identity_block(out, channels=chanels_per_block[k])
                
    # Final layers
    out = IntGlobalAveragePooling2D()(out)

    # Final Dense layers
    out = IntDense(num_classes, activation=None)(out)
    out = IntBatchNormalization()(out)
    if predict_mod == True:
        outputs = Activation(IntSoftMax)(out)
    else:
        outputs = Activation(IntSoftMaxSingle)(out)

    model = keras.Model(inputs, outputs, name='INNResNet50')

    return model

    

    