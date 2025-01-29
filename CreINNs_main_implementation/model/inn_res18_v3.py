from model.inn_layers_v3 import IntDense, IntConv2D, IntFlatten, IntBatchNormalization, IntAdd
from model.inn_layers_v3 import IntDropout, IntAveragePooling2D, IntMaxPooling2D, IntGlobalAveragePooling2D, IntZeroPadding2D
from model.inn_layers_v3 import IntRelu, IntSoftMaxSingle, IntSoftMax, IntSigMoidSingle, IntSigMoid
from keras.models import Model
from keras.layers import Activation
from keras import Input
import tensorflow as tf
from tensorflow import keras
import numpy as np

INIT_SCHEME ='glorot_uniform'
INIT_SCHEME_2 = 'glorot_uniform'
USB = True
# USB = False

def inn_resnet18(input_shape, num_classes, predict_mod):
  """Constructs a ResNet18 model.
  
  Returns:
    keras.Model.
  """

  filters = [64, 128, 256, 512]
  kernels = [(3, 3), (3, 3), (3, 3), (3, 3)]
  strides = [(1, 1), (2, 2), (2, 2), (2, 2)]

  image = [Input(input_shape), Input(input_shape)]
    
  x = IntConv2D(
      64,
      (3, 3),
      strides=(1, 1),
      padding='same',
      center_kernel_initializer=INIT_SCHEME, 
      radius_kernel_initializer=INIT_SCHEME_2, use_bias=USB)(image)

  for i in range(len(kernels)):
    x = _resnet_block(
        x,
        filters[i],
        kernels[i],
        strides[i],
        )

  x = IntBatchNormalization()(x)
  x = IntRelu()(x)
  x = IntAveragePooling2D(4, 1)(x)
  x = IntFlatten()(x)

  x = IntDense(num_classes)(x)
  x = IntBatchNormalization()(x)
  if predict_mod == True:
      if num_classes>=2:
        x = Activation(IntSoftMax)(x)
      else:
        x = Activation(IntSigMoid)(x)
  else:
      if num_classes>=2:
        x = Activation(IntSoftMaxSingle)(x)
      else:
        x = Activation(IntSigMoidSingle)(x)

  model = keras.Model(inputs=image, outputs=x, name='INNresnet18')
  return model


def _resnet_block(x, filters, kernel, stride):
  """Network block for ResNet."""
  x = IntBatchNormalization()(x)
  x = IntRelu()(x)

  if stride != 1 or filters != x.shape[1]:
    shortcut = _projection_shortcut(x, filters, stride)
  else:
    shortcut = x

  x = IntConv2D(
      filters,
      kernel,
      strides=stride,
      padding='same',
      center_kernel_initializer=INIT_SCHEME, 
      radius_kernel_initializer=INIT_SCHEME_2, use_bias=USB)(x)
    
  x = IntBatchNormalization()(x)
  x = IntRelu()(x)

  x = IntConv2D(
      filters,
      kernel,
      strides=1,
      padding='same', 
      center_kernel_initializer=INIT_SCHEME, 
      radius_kernel_initializer=INIT_SCHEME_2, use_bias=USB)(x)

  x = IntAdd()([x, shortcut])
  return x


def _projection_shortcut(x, out_filters, stride):
  x = IntConv2D(
      out_filters,
      1,
      strides=stride,
      padding='valid',
      center_kernel_initializer=INIT_SCHEME, 
      radius_kernel_initializer=INIT_SCHEME_2, use_bias=USB)(x)
  return x