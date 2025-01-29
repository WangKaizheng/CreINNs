from keras.models import Model
from keras.layers import Activation, Conv2D, BatchNormalization, AveragePooling2D, Dense, Flatten, Add
from keras import Input
import tensorflow as tf
from tensorflow import keras
import numpy as np

# USB = False
USB = True

def resnet18(input_shape, num_classes=10):
  """Constructs a ResNet18 model.
  
  Returns:
    keras.Model.
  """

  filters = [64, 128, 256, 512]
  kernels = [(3, 3), (3, 3), (3, 3), (3, 3)]
  strides = [(1, 1), (2, 2), (2, 2), (2, 2)]

  image = Input(input_shape)
    
  x = Conv2D(
      64,
      (3, 3),
      strides=(1, 1),
      padding='same',
      use_bias=USB)(image)

  for i in range(len(kernels)):
    x = _resnet_block(
        x,
        filters[i],
        kernels[i],
        strides[i],
        )

  x = BatchNormalization()(x)
  x = tf.nn.relu(x)
  x = AveragePooling2D(4, 1)(x)
  x = Flatten()(x)

  x = Dense(num_classes)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  if num_classes>=2:
    x = Activation('softmax')(x)
  else:
    x = Activation('sigmoid')(x)

  model = keras.Model(inputs=image, outputs=x, name='Resnet18')
  return model


def _resnet_block(x, filters, kernel, stride):
  """Network block for ResNet."""
  x = BatchNormalization()(x)
  x = tf.nn.relu(x)

  if stride != 1 or filters != x.shape[1]:
    shortcut = _projection_shortcut(x, filters, stride)
  else:
    shortcut = x

  x = Conv2D(
      filters,
      kernel,
      strides=stride,
      padding='same',
      use_bias=USB)(x)
    
  x = BatchNormalization()(x)
  x = tf.nn.relu(x)

  x = Conv2D(
      filters,
      kernel,
      strides=1,
      padding='same', 
      use_bias=USB)(x)

  x = Add()([x, shortcut])
  return x


def _projection_shortcut(x, out_filters, stride):
  x = Conv2D(
      out_filters,
      1,
      strides=stride,
      padding='valid',
      use_bias=USB)(x)
  return x