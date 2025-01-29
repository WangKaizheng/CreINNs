import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import activations
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import constraints
from tensorflow.python.framework import tensor_shape
import numpy as np


@keras.saving.register_keras_serializable(package="my_package", name="IntDense")
class IntDense(keras.layers.Layer):
    def __init__(
            self,
            units,
            activation=None,
            center_kernel_initializer='glorot_uniform',
            radius_kernel_initializer='glorot_normal',
            center_bias_initializer='zeros',
            radius_bias_initializer='zeros',
            use_bias=True,
            center_kernel_regularizer=None,
            radius_kernel_regularizer=None,
            center_bias_regularizer=None,
            radius_bias_regularizer=None,
            positive_input=True,
            **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(IntDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.center_kernel_initializer = initializers.get(center_kernel_initializer)
        self.radius_kernel_initializer = initializers.get(radius_kernel_initializer)
        self.center_bias_initializer = initializers.get(center_bias_initializer)
        self.radius_bias_initializer = initializers.get(radius_bias_initializer)
        self.center_kernel_regularizer = regularizers.get(center_kernel_regularizer)
        self.radius_kernel_regularizer = regularizers.get(radius_kernel_regularizer)
        self.center_bias_regularizer = regularizers.get(center_bias_regularizer)
        self.radius_bias_regularizer = regularizers.get(radius_bias_regularizer)
        self.use_bias = use_bias
        self.positive_input = positive_input

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_dim = input_shape[0][-1]
        else:
            input_dim = input_shape[-1]

        self.center_kernel = self.add_weight(
            name='center_kernel',
            shape=(input_dim, self.units),
            initializer=self.center_kernel_initializer,
            regularizer=self.center_kernel_regularizer,
            trainable=True,
        )

        self.radius_kernel = self.add_weight(
            name='radius_kernel',
            shape=(input_dim, self.units),
            initializer=self.radius_kernel_initializer,
            regularizer=self.radius_kernel_regularizer,
            constraint=constraints.get('nonneg'),
            trainable=True,
        )

        if self.use_bias:

            self.center_bias = self.add_weight(
                name='center_bias',
                shape=(self.units,),
                initializer=self.center_bias_initializer,
                regularizer=self.center_bias_regularizer,
                trainable=True,
            )
            self.radius_bias = self.add_weight(
                name='radius_bias',
                shape=(self.units,),
                initializer=self.radius_bias_initializer,
                regularizer=self.radius_bias_regularizer,
                constraint=constraints.get('nonneg'),
                trainable=True,
            )

        else:
            self.center_bias = None
            self.radius_bias = None

    def call(self, inputs):

        lo, hi = inputs

        if self.positive_input:
            lo_out = K.dot(lo, K.maximum(self.center_kernel-self.radius_kernel, 0.0)) + K.dot(
                hi, K.minimum(self.center_kernel-self.radius_kernel, 0.0)
            )
            hi_out = K.dot(lo, K.minimum(self.center_kernel+self.radius_kernel, 0.0)) + K.dot(
                hi, K.maximum(self.center_kernel+self.radius_kernel, 0.0)
            )
        else:
            lo_out = K.dot(K.minimum(hi, 0.0), K.minimum(self.center_kernel+self.radius_kernel, 0.0)) + K.dot(
                K.maximum(lo, 0.0), K.maximum(self.center_kernel-self.radius_kernel, 0.0)) + K.dot(
                K.maximum(hi, 0.0), K.minimum(self.center_kernel-self.radius_kernel, 0.0)) + K.minimum(
                K.dot(K.minimum(lo, 0.0), K.maximum(self.center_kernel+self.radius_kernel, 0.0)) - K.dot(K.maximum(hi, 0.0),
                                                                                   K.minimum(self.center_kernel-self.radius_kernel, 0.0)), 0.0
            )
            hi_out = K.dot(K.maximum(lo, 0.0), K.minimum(self.center_kernel+self.radius_kernel, 0.0)) + K.dot(
                K.minimum(hi, 0.0), K.maximum(self.center_kernel-self.radius_kernel, 0.0)) + K.dot(
                K.maximum(hi, 0.0), K.maximum(self.center_kernel+self.radius_kernel, 0.0)) + K.maximum(
                K.dot(K.minimum(lo, 0.0), K.minimum(self.center_kernel-self.radius_kernel, 0.0)) - K.dot(K.maximum(hi, 0.0),
                                                                                   K.maximum(self.center_kernel+self.radius_kernel, 0.0)), 0.0
            )

        if self.use_bias:
            lo_out = K.bias_add(
                lo_out,
                self.center_bias-self.radius_bias,
                data_format="channels_last",
            )
            hi_out = K.bias_add(
                hi_out,
                self.center_bias+self.radius_bias,
                data_format="channels_last",
            )

        if self.activation is not None:
            lo_out = self.activation(lo_out)
            hi_out = self.activation(hi_out)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntDense, self).get_config()
        config.update(
            {
                'units': self.units,
                'activation': activations.serialize(self.activation),
                'center_kernel_initializer': initializers.serialize(self.center_kernel_initializer),
                'radius_kernel_initializer': initializers.serialize(self.radius_kernel_initializer),
                'center_bias_initializer': initializers.serialize(self.center_bias_initializer),
                'radius_bias_initializer': initializers.serialize(self.radius_bias_initializer),
                'center_kernel_regularizer': regularizers.serialize(self.center_kernel_regularizer),
                'radius_kernel_regularizer': regularizers.serialize(self.radius_kernel_regularizer),
                'center_bias_regularizer': regularizers.serialize(self.center_bias_regularizer),
                'radius_bias_regularizer': regularizers.serialize(self.radius_bias_regularizer),
                'use_bias': self.use_bias,
                'positive_inputs': self.positive_input
            }
        )
        return config

@keras.saving.register_keras_serializable(package="my_package", name="IntConv2D")
class IntConv2D(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 data_format=None,
                 activation=None,
                 center_kernel_initializer='glorot_uniform',
                 radius_kernel_initializer='glorot_normal',
                 center_bias_initializer='zeros',
                 radius_bias_initializer='zeros',
                 center_kernel_regularizer=None,
                 radius_kernel_regularizer=None,
                 center_bias_regularizer=None,
                 radius_bias_regularizer=None,
                 use_bias=True,
                 positive_input=True,
                 ):
        super(IntConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.activation = activations.get(activation)
        self.center_kernel_initializer = initializers.get(center_kernel_initializer)
        self.radius_kernel_initializer = initializers.get(radius_kernel_initializer)
        self.center_bias_initializer = initializers.get(center_bias_initializer)
        self.radius_bias_initializer = initializers.get(radius_bias_initializer)
        self.center_kernel_regularizer = regularizers.get(center_kernel_regularizer)
        self.radius_kernel_regularizer = regularizers.get(radius_kernel_regularizer)
        self.center_bias_regularizer = regularizers.get(center_bias_regularizer)
        self.radius_bias_regularizer = regularizers.get(radius_bias_regularizer)
        self.use_bias = use_bias
        self.positive_input = positive_input

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.center_kernel = self.add_weight(
            name='center_kernel',
            shape=kernel_shape,
            initializer=self.center_kernel_initializer,
            regularizer=self.center_kernel_regularizer,
            trainable=True,
        )

        self.radius_kernel = self.add_weight(
            name='radius_kernel',
            shape=kernel_shape,
            initializer=self.radius_kernel_initializer,
            regularizer=self.radius_kernel_regularizer,
            constraint=constraints.get('nonneg'),
            trainable=True,
        )

        if self.use_bias:
            self.center_bias = self.add_weight(
                name='center_bias',
                shape=(self.filters,),
                initializer=self.center_bias_initializer,
                regularizer=self.center_bias_regularizer,
                trainable=True,
            )

            self.radius_bias = self.add_weight(
                name='radius_bias',
                shape=(self.filters,),
                initializer=self.radius_bias_initializer,
                regularizer=self.radius_bias_regularizer,
                constraint=constraints.get('nonneg'),
                trainable=True
            )
        else:
            self.center_bias = None
            self.radius_bias = None

    def call(self, inputs):
        @tf.function
        def convfunc(x, k):
            out = K.conv2d(
                x, k, strides=self.strides, padding=self.padding, data_format=self.data_format, dilation_rate=None
            )
            return out

        lo, hi = inputs
        
        min_kernel_pos = K.maximum(self.center_kernel-self.radius_kernel, 0.0)
        min_kernel_neg = K.minimum(self.center_kernel-self.radius_kernel, 0.0)
        max_kernel_pos = K.maximum(self.center_kernel+self.radius_kernel, 0.0)
        max_kernel_neg = K.minimum(self.center_kernel+self.radius_kernel, 0.0)

        if self.positive_input:
            lo_out = (convfunc(lo, min_kernel_pos) + convfunc(hi, min_kernel_neg))
            hi_out = (convfunc(lo, max_kernel_neg) + convfunc(hi, max_kernel_pos))
        else:
            lo_neg = K.minimum(lo, 0.0)
            lo_pos = K.maximum(lo, 0.0)
            hi_neg = K.minimum(hi, 0.0)
            hi_pos = K.maximum(hi, 0.0)

            lo_out = convfunc(hi_neg, max_kernel_neg) + convfunc(lo_pos, min_kernel_pos) + K.minimum(
                convfunc(lo_neg, max_kernel_pos) - convfunc(hi_pos, min_kernel_neg), 0.0
            ) + convfunc(hi_pos, min_kernel_neg)

            hi_out = convfunc(lo_pos, max_kernel_neg) + convfunc(hi_neg, min_kernel_pos) + K.maximum(
                convfunc(lo_neg, min_kernel_neg) - convfunc(hi_pos, max_kernel_pos), 0.0
            ) + convfunc(hi_pos, max_kernel_pos)

        if self.use_bias:
            lo_out = K.bias_add(
                lo_out,
                self.center_bias-self.radius_bias,
                data_format="channels_last",
            )
            hi_out = K.bias_add(
                hi_out,
                self.center_bias+self.radius_bias,
                data_format="channels_last",
            )

        if self.activation is not None:
            lo_out = self.activation(lo_out)
            hi_out = self.activation(hi_out)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'center_kernel_initializer': initializers.serialize(self.center_kernel_initializer),
            'radius_kernel_initializer': initializers.serialize(self.radius_kernel_initializer),
            'center_bias_initializer': initializers.serialize(self.center_bias_initializer),
            'radius_bias_initializer': initializers.serialize(self.radius_bias_initializer),
            'center_kernel_regularizer': regularizers.serialize(self.center_kernel_regularizer),
            'radius_kernel_regularizer': regularizers.serialize(self.radius_kernel_regularizer),
            'center_bias_regularizer': regularizers.serialize(self.center_bias_regularizer),
            'radius_bias_regularizer': regularizers.serialize(self.radius_bias_regularizer),
            'use_bias': self.use_bias,
            'positive_inputs': self.positive_input
        }
        )
        return config

@keras.saving.register_keras_serializable(package="my_package", name="IntFlatten")
class IntFlatten(keras.layers.Layer):

    def __init__(self,
                 data_format=None
                 ):
        super(IntFlatten, self).__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)

    def call(self, inputs):
        lo, hi = inputs

        single_flatten = tf.keras.layers.Flatten(data_format=self.data_format)

        lo_out = single_flatten(lo)
        hi_out = single_flatten(hi)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntFlatten, self).get_config()
        config.update(
            {
                'data_format': self.data_format,
            }
        )

        return config

@keras.saving.register_keras_serializable(package="my_package", name="IntBatchNormalization")
class IntBatchNormalization(keras.layers.Layer):
    def __init__(
            self,
            axis=-1,
            momentum = 0.999,
            epsilon=1e-3,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            trainable=True,
            **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super(IntBatchNormalization, self).__init__(**kwargs)
        if isinstance(axis, (list, tuple)):
          self.axis = axis[:]
        elif isinstance(axis, int):
          self.axis = axis
        else:
          raise TypeError('Expected an int or a list/tuple of ints for the '
                      'argument \'axis\', but received: %r' % axis)
        self.momentum = momentum
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.trainable = trainable

    def build(self, input_shape_list):
        input_shape = tensor_shape.TensorShape(input_shape_list[0])
        if not input_shape.ndims:
            raise ValueError('Input has undefined rank:', input_shape)
        ndims = len(input_shape)

        if isinstance(self.axis, int):
           self.axis = [self.axis]

        for idx, x in enumerate(self.axis):
            if x < 0:
               self.axis[idx] = ndims + x

        # Validate axes
        for x in self.axis:
           if x < 0 or x >= ndims:
              raise ValueError('Invalid axis: %d' % x)
        if len(self.axis) != len(set(self.axis)):
           raise ValueError('Duplicate axis: %s' % self.axis)

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
           if axis_to_dim[x] is None:
              raise ValueError('Input has undefined `axis` dimension. Input shape: ',
                            input_shape)


        if len(axis_to_dim) == 1:
          # Single axis batch norm (most common/default use-case)
          param_shape = (list(axis_to_dim.values())[0],)
        else:
          # Parameter shape is the original shape but with 1 in all non-axis dims
          param_shape = [axis_to_dim[i] if i in axis_to_dim
                        else 1 for i in range(ndims)]
          if self.virtual_batch_size is not None:
            # When using virtual batches, add an extra dim at index 1
            param_shape.insert(1, 1)
            for idx, x in enumerate(self.axis):
              self.axis[idx] = x + 1      # Account for added dimension

        self.center_gamma = self.add_weight(
            name='center_gamma',
            shape=param_shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            trainable=True,
            experimental_autocast=False
            )
        self.radius_gamma = self.add_weight(
            name='radius_gamma',
            shape=param_shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            trainable=True,
            experimental_autocast=False
            )
        
        self.center_beta = self.add_weight(
            name='center_beta',
            shape=param_shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            trainable=True,
            experimental_autocast=False
            )
        
        self.radius_beta = self.add_weight(
              name='raduis_beta',
              shape=param_shape,
              initializer=self.beta_initializer,
              regularizer=self.beta_regularizer,
              constraint=self.beta_constraint,
              trainable=True,
              experimental_autocast=False
              )
    
        self.center_moving_mean = self.add_weight(
            name='center_moving_mean',
            shape=param_shape,
            initializer=self.moving_mean_initializer,
            trainable=False,
            )
        
        self.radius_moving_mean = self.add_weight(
            name='radius_moving_mean',
            shape=param_shape,
            initializer=self.moving_mean_initializer,
            trainable=False,
            )
        
        self.center_moving_variance = self.add_weight(
            name='center_moving_variance',
            shape=param_shape,
            initializer=self.moving_variance_initializer,
            trainable=False,
            )
        
        self.radius_moving_variance = self.add_weight(
            name='radius_moving_variance',
            shape=param_shape,
            initializer=self.moving_variance_initializer,
            trainable=False,
            )
        

    def call(self, inputs):

        lo, hi = inputs

        center = (lo + hi) / 2
        radius  = (hi - lo) / 2


        if self.trainable:
           if len(center.shape) == 2:
              center_mean, center_var = tf.nn.moments(center, [0])
              radius_mean, radius_var = tf.nn.moments(radius, [0])
           else:
              center_mean, center_var = tf.nn.moments(center, [0, 1, 2])
              radius_mean, radius_var = tf.nn.moments(radius, [0, 1, 2])

           train_mean_center = tf.compat.v1.assign(
               self.center_moving_mean, self.center_moving_mean * self.momentum + center_mean * (1 - self.momentum))
           train_var_center = tf.compat.v1.assign(
               self.center_moving_variance, self.center_moving_variance * self.momentum + center_var * (1 - self.momentum))
           
           train_mean_radius = tf.compat.v1.assign(
               self.radius_moving_mean, self.radius_moving_mean * self.momentum + radius_mean * (1 - self.momentum))
           train_var_radius = tf.compat.v1.assign(
               self.radius_moving_variance, self.radius_moving_variance * self.momentum + radius_var * (1 - self.momentum))
           
           with tf.control_dependencies([train_mean_center, train_var_center, train_mean_radius, train_var_radius]):
              center_scaled = tf.nn.batch_normalization(center, center_mean, center_var, self.center_beta, self.center_gamma, self.epsilon)
              radius_scaled = tf.nn.batch_normalization(radius, radius_mean, radius_var, self.radius_beta, self.radius_gamma, self.epsilon)
        else:
           center_scaled = tf.nn.batch_normalization(center, self.center_moving_mean, self.center_moving_variance, self.center_beta, self.center_gamma, self.epsilon)
           radius_scaled = tf.nn.batch_normalization(radius, self.radius_moving_mean, self.radius_moving_variance, self.radius_beta, self.radius_gamma, self.epsilon)

        lo_out = center_scaled - tf.math.abs(radius_scaled)
        hi_out = center_scaled + tf.math.abs(radius_scaled)

        output = [lo_out, hi_out]

        return output
    def get_config(self):
        config = super(SimIntBatchNormalization, self).get_config()
        config.update({
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint),
            'trainable': self.trainable
        }
        )
        return config

@keras.saving.register_keras_serializable(package="my_package", name="IntDropout")
class IntDropout(keras.layers.Layer):
    """Applies Dropout to INN.
    # Reference
        - [Oala, L., Heiß, C., Macdonald, J., M  ̈arz, M., Kutyniok, G., and Samek, W. 
        Detecting failure modes in image reconstructions with interval neural network uncertainty.
        International Journal of Computer Assisted Radiology and Surgery, 16(12):2089–2097, 2021].
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(IntDropout, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        if isinstance(inputs, list):
            raise ValueError('Noise shape issue')
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            if self.seed is None:
                seed = np.random.randint(10e6)
                self.seed = seed
            else:
                seed = self.seed

            if isinstance(inputs, list):
                noise_shape = self._get_noise_shape(inputs[0])
                dropped_inputs = [K.dropout(a, self.rate, noise_shape, seed=seed) for a in inputs]
                return [
                    K.in_train_phase(a, b, training=training) for a,b in zip(dropped_inputs, inputs)
                ]
            else:
                noise_shape = self._get_noise_shape(inputs)
                dropped_inputs = K.dropout(inputs, self.rate, noise_shape, seed=seed)
                return K.in_train_phase(dropped_inputs, inputs,
                                        training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,
                  'noise_shape': self.noise_shape,
                  'seed': self.seed}
        base_config = super(IntDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


@keras.saving.register_keras_serializable(package="my_package", name="IntAveragePooling2D")
class IntAveragePooling2D(keras.layers.Layer):

    def __init__(self,
                 pool_size = (2, 2),
                 strides = None,
                 padding = 'valid',
                 data_format=None,
                 **kwargs
                 ):
        super(IntAveragePooling2D, self).__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        lo, hi = inputs

        lo_out = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.strides,
                                              padding=self.padding, data_format=self.data_format)(lo)
        
        hi_out = tf.keras.layers.AveragePooling2D(pool_size=self.pool_size, strides=self.strides,
                                              padding=self.padding, data_format=self.data_format)(hi)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntAveragePooling2D, self).get_config()
        config.update(
            {
                'pool_size': self.pool_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
            }
        )

        return config

@keras.saving.register_keras_serializable(package="my_package", name="IntRelu")
class IntRelu(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        super(IntRelu, self).__init__()

    def call(self, inputs):
        lo, hi = inputs
        

        lo_out = tf.nn.relu(lo)
        
        hi_out = tf.nn.relu(hi)

        output = [lo_out, hi_out]

        return output

@keras.saving.register_keras_serializable(package="my_package", name="IntMaxPooling2D")
class IntMaxPooling2D(keras.layers.Layer):

    def __init__(self,
                 pool_size = (2, 2),
                 strides = None,
                 padding = 'valid',
                 data_format=None,
                 **kwargs
                 ):
        super(IntMaxPooling2D, self).__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        lo, hi = inputs

        lo_out = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides,
                                              padding=self.padding, data_format=self.data_format)(lo)
        
        hi_out = tf.keras.layers.MaxPooling2D(pool_size=self.pool_size, strides=self.strides,
                                              padding=self.padding, data_format=self.data_format)(hi)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntMaxPooling2D, self).get_config()
        config.update(
            {
                'pool_size': self.pool_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
            }
        )

        return config


@keras.saving.register_keras_serializable(package="my_package", name="IntGlobalAveragePooling2D")
class IntGlobalAveragePooling2D(keras.layers.Layer):

    def __init__(self,
                 data_format=None,
                 keepdims=False,
                 **kwargs
                 ):
        
        super(IntGlobalAveragePooling2D, self).__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.keepdims = keepdims

    def call(self, inputs):
        lo, hi = inputs

        lo_out = tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, keepdims=self.keepdims)(lo)
        
        hi_out = tf.keras.layers.GlobalAveragePooling2D(data_format=self.data_format, keepdims=self.keepdims)(hi)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntGlobalAveragePooling2D, self).get_config()
        config.update(
            {
                'keepdims': self.keepdims,
                'data_format': self.data_format,
            }
        )

        return config


@keras.saving.register_keras_serializable(package="my_package", name="IntZeroPadding2D")
class IntZeroPadding2D(keras.layers.Layer):

    def __init__(self,
                 padding=(1, 1),
                 data_format=None,
                 **kwargs
                 ):
        
        super(IntZeroPadding2D, self).__init__()
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.padding = padding

    def call(self, inputs):
        lo, hi = inputs

        lo_out = tf.keras.layers.ZeroPadding2D(padding=self.padding, data_format=self.data_format)(lo)
        
        hi_out = tf.keras.layers.ZeroPadding2D(padding=self.padding, data_format=self.data_format)(hi)

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntZeroPadding2D, self).get_config()
        config.update(
            {
                'padding': self.padding,
                'data_format': self.data_format,
            }
        )

        return config


@keras.saving.register_keras_serializable(package="my_package", name="IntAdd")
class IntAdd(keras.layers.Layer):

    def __init__(self,
                 **kwargs
                 ):
        
        super(IntAdd, self).__init__()

    def call(self, inputs_list):
        
        inputs1, inputs2 = inputs_list
        
        lo1, hi1 = inputs1

        lo2, hi2 = inputs2

        lo_out = tf.keras.layers.Add()([lo1, lo2])
        
        hi_out = tf.keras.layers.Add()([hi1, hi2])

        output = [lo_out, hi_out]

        return output

    def get_config(self):
        config = super(IntAdd, self).get_config()

        return config
        
####################### Activation Functions ###################################
@keras.saving.register_keras_serializable(package="my_package", name="IntLeakyReLu")
@tf.function
def IntLeakyReLu(inputs):

    leakyReLU = tf.keras.layers.LeakyReLU(alpha=0.1)
    lo, hi = inputs

    lo_out = leakyReLU(lo)
    hi_out = leakyReLU(hi)

    output = [lo_out, hi_out]

    return output

@keras.saving.register_keras_serializable(package="my_package", name="IntReLu")
@tf.function
def IntReLu(inputs):

    lo, hi = inputs

    lo_out = tf.nn.relu(lo)
    hi_out = tf.nn.relu(hi)
    # lo_out = lo
    # hi_out = hi
    output = [lo_out, hi_out]

    return output
    
@keras.saving.register_keras_serializable(package="my_package", name="IntSoftMax")
@tf.function
def IntSoftMax(inputs):
  
    lo, hi = inputs

    lo_out = K.exp(lo) / (K.sum(K.exp(0.5 * lo + 0.5 * hi), axis=-1, keepdims=True) - K.exp(0.5 * lo + 0.5 * hi) + K.exp(lo))
    hi_out = K.exp(hi) / ((K.sum(K.exp(0.5 * lo + 0.5 * hi), axis=-1, keepdims=True) - K.exp(0.5 * lo + 0.5 * hi)) + K.exp(hi) )
    # output = [lo_out, hi_out]

    # lo_out = tf.clip_by_value(lo_out, 1e-9, 0.999999)
    # hi_out = tf.clip_by_value(hi_out, 1e-9, 0.999999)
    lo_out_mod = K.maximum(lo_out, 1.0-(K.sum(hi_out, axis=-1, keepdims=1)-hi_out))
    hi_out_mod = K.minimum(hi_out, 1.0-(K.sum(lo_out, axis=-1, keepdims=1)-lo_out))
    output = [lo_out_mod, hi_out_mod]

    return output


@keras.saving.register_keras_serializable(package="my_package", name="IntSoftMaxSingle")
@tf.function
def IntSoftMaxSingle(inputs):
  
    lo, hi = inputs

    lo_out = K.exp(lo) / (K.sum(K.exp(0.5 * lo + 0.5 * hi), axis=-1, keepdims=True) - K.exp(0.5 * lo + 0.5 * hi) + K.exp(lo))
    hi_out = K.exp(hi) / ((K.sum(K.exp(0.5 * lo + 0.5 * hi), axis=-1, keepdims=True) - K.exp(0.5 * lo + 0.5 * hi)) + K.exp(hi) )

    lo_out_mod = K.maximum(lo_out, 1.0-(K.sum(hi_out, axis=-1, keepdims=1)-hi_out))
    hi_out_mod = K.minimum(hi_out, 1.0-(K.sum(lo_out, axis=-1, keepdims=1)-lo_out))

    alpha_num = 1.0 - K.sum(lo_out_mod, axis=-1, keepdims=True)
    alpha_denom = K.sum(hi_out_mod-lo_out_mod, axis=-1, keepdims=True)
    
    alpha = alpha_num/alpha_denom
    
    output = hi_out_mod * alpha + (1.0-alpha) * lo_out_mod

    # alpha_num = 1.0 - K.sum(lo_out, axis=-1, keepdims=True)
    # alpha_denom = K.sum(hi_out-lo_out, axis=-1, keepdims=True)
    
    # alpha = alpha_num/alpha_denom
    
    # output = hi_out * alpha + (1.0-alpha) * lo_out

    return output


# def reachable_probability_intervals(lower_probs, upper_probs):
#     lower_probs_mod = np.maximum(lower_probs, 1.0-(np.sum(upper_probs, axis=-1, keepdims=1)-upper_probs))
#     upper_probs_mod = np.minimum(upper_probs, 1.0-(np.sum(lower_probs, axis=-1, keepdims=1)-lower_probs))
#     return lower_probs_mod, upper_probs_mod


@keras.saving.register_keras_serializable(package="my_package", name="IntSigMoid")
@tf.function
def IntSigMoid(inputs):
  
    lo, hi = inputs
    lo_out_mod = tf.math.sigmoid(lo)
    hi_out_mod = tf.math.sigmoid(hi)
    output = [lo_out_mod, hi_out_mod]

    return output


@keras.saving.register_keras_serializable(package="my_package", name="IntSigMoidSingle")
@tf.function
def IntSigMoidSingle(inputs):
  
    lo, hi = inputs

    lo_out_mod = tf.math.sigmoid(lo)
    hi_out_mod = tf.math.sigmoid(hi)
    
    output = 0.5 * (lo_out_mod + hi_out_mod)
    
    return output