"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4

For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

The weights for all 16 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381) (CVPR 2018)

Tests comparing this model to the existing Tensorflow model can be
found at [mobilenet_v2_keras]
(https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np
import keras
import tensorflow as tf
from keras_applications import imagenet_utils as utils
from keras_applications import correct_pad
from keras.layers import Lambda
import PIL.Image
from scipy.io import loadmat
from scipy import stats
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import pickle
import geopandas 
import shapely
import pandas as pd
import time
import math
import gc
import glob
import sys
from re import split

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Cha

def preprocess_input(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return utils.preprocess_input(x, mode='tf', **kwargs)


#Function to pre-process the input image to ensure uniform size and color
def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode='rgb', out=None):
    """
    Consistent preprocessing of images batches
    :param image_paths: iterable: images to process
    :param crop_size: tuple: crop images if specified
    :param img_size: tuple: resize images if specified
    :param color_mode: Use rgb or change to bgr mode based on type of model you want to use
    :param out: append output to this iterable if specified
    """
    img_list = []

    for im_path in image_paths:
        size = 224
        ret = PIL.Image.open(im_path)
        ret = ret.resize((size, size))
        ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
        if ret.ndim == 2:
            ret.resize((size, size, 1))
            ret = np.repeat(ret, 3, axis=-1)
        #ret = ret.transpose((2, 0, 1))
        #ret = np.flip(ret,0)
        global backend
        x = preprocess_input(ret, 
            data_format=backend.image_data_format())
        img_list.append(x)


    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        print(im_path)
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def MobileNetV2(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000,
                lambda_mask = None,
                **kwargs):
    """Instantiates the MobileNetV2 architecture.

    # Arguments
        input_shape: optional shape tuple, to be specified if you would
            like to use a model with an input img resolution that is not
            (224, 224, 3).
            It should have exactly 3 inputs channels (224, 224, 3).
            You can also omit this option if you would like
            to infer input_shape from an input_tensor.
            If you choose to include both input_tensor and input_shape then
            input_shape will be used if they match, if the shapes
            do not match then we will throw an error.
            E.g. `(160, 160, 3)` would be one valid value.
        alpha: controls the width of the network. This is known as the
        width multiplier in the MobileNetV2 paper, but the name is kept for
        consistency with MobileNetV1 in Keras.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape or invalid alpha, rows when
            weights='imagenet'
    """
    global backend, layers, models, keras_utils, debug
    debug = False
    backend= keras.backend
    layers = keras.layers
    models = keras.models
    keras_utils = keras.utils
 
    # TODO Change path to v1.1
    BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                        'releases/download/v1.1/')

    backend= keras.backend
    layers = keras.layers
    models = keras.models
    keras_utils = keras.utils

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    # If both input_shape and input_tensor are used, they should match
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(
                    keras_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor,
                                 'is not type input_tensor')
        if is_input_t_tensor:
            if backend.image_data_format == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError('input_shape: ', input_shape,
                                     'and input_tensor: ', input_tensor,
                                     'do not meet the same shape requirements')
        else:
            raise ValueError('input_tensor specified: ', input_tensor,
                             'is not a keras tensor')

    # If input_shape is None, infer shape from input_tensor
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor,
                             'is type: ', type(input_tensor),
                             'which is not a valid type')

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = utils._obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of `0.35`, `0.50`, `0.75`, '
                             '`1.0`, `1.3` or `1.4` only.')

        if rows != cols or rows not in [96, 128, 160, 192, 224]:
            rows = 224
            warnings.warn('`input_shape` is undefined or non-square, '
                          'or `rows` is not in [96, 128, 160, 192, 224].'
                          ' Weights for input shape (224, 224) will be'
                          ' loaded as the default.')

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    global start_index, end_index
    start_index = end_index = 0
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (default_size//2 * default_size//2 * first_block_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (default_size//2, default_size//2, first_block_filters))
        if debug:
            print('Conv1',start_index,end_index)
    else:
        x_mask = np.ones(shape=((default_size//2, default_size//2, first_block_filters)))

    x_mask  = backend.variable(x_mask) #Numpy array to Tensor
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (default_size//2 * default_size//2 * first_block_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (default_size//2, default_size//2, first_block_filters))
        if debug:
            print('bn_Conv1',start_index,end_index)
    else:
        x_mask = np.ones(shape=((default_size//2, default_size//2, first_block_filters)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14, lambda_mask=lambda_mask)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15, lambda_mask=lambda_mask)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16, lambda_mask=lambda_mask)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (x.shape[1] * x.shape[2]* last_block_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (x.shape[1], x.shape[2], last_block_filters))
        if debug:
            print('Conv_1',start_index,end_index)
    else:
        x_mask = np.ones(shape=((x.shape[1], x.shape[2], last_block_filters)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (x.shape[1] * x.shape[2] * last_block_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (x.shape[1], x.shape[2], last_block_filters))
        if debug:
            print('Conv_1_bn',start_index,end_index)
    else:
        x_mask = np.ones(shape=((x.shape[1],x.shape[2], last_block_filters)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.ReLU(6., name='out_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax', use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(model_name, weight_path, cache_subdir='models')
        else:
            model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' + str(alpha) + '_' + str(rows) + '_no_top' + '.h5')
            weight_path = BASE_WEIGHT_PATH + model_name
            weights_path = tf.keras.utils.get_file(model_name, weight_path, cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, lambda_mask = None):
    global debug
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    #print(prefix, inputs.shape,inputs.shape[0],inputs.shape[1] , in_channels, pointwise_conv_filters, pointwise_filters, filters)
    global start_index, end_index
    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        #################
        if lambda_mask is not None:
            start_index = end_index
            end_index = start_index + (inputs.shape[1] * inputs.shape[2] * inputs.shape[3]*expansion)
            x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1], inputs.shape[2], inputs.shape[3]*expansion))
            if debug:
                print(prefix + 'expand',start_index,end_index)
        else:
            x_mask = np.ones(shape=((inputs.shape[1], inputs.shape[2], inputs.shape[3]*expansion)))

        x_mask  = backend.variable(x_mask)
        x = Lambda(lambda z: z * x_mask)(x)
        ####################
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
  
        #################
        if lambda_mask is not None:
            start_index = end_index
            end_index = start_index + (inputs.shape[1] * inputs.shape[2] * inputs.shape[3]*expansion)
            x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1], inputs.shape[2], inputs.shape[3]*expansion))
            if debug:
                print(prefix + 'expand_BN',start_index,end_index)
        else:
            x_mask = np.ones(shape=((inputs.shape[1], inputs.shape[2], inputs.shape[3]*expansion)))

        x_mask  = backend.variable(x_mask)
        x = Lambda(lambda z: z * x_mask)(x)
        ####################
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3), name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,padding='same' if stride == 1 else 'valid', name=prefix + 'depthwise')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (inputs.shape[1]//stride * inputs.shape[2]//stride * inputs.shape[3]*expansion)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1]//stride, inputs.shape[2]//stride, inputs.shape[3]*expansion))
        if debug:
            print(prefix + 'depthwise',start_index,end_index)
    else:
        x_mask = np.ones(shape=((inputs.shape[1]//stride, inputs.shape[2]//stride, inputs.shape[3]*expansion)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.BatchNormalization(axis=channel_axis,epsilon=1e-3,momentum=0.999, name=prefix + 'depthwise_BN')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (inputs.shape[1]//stride * inputs.shape[2]//stride * inputs.shape[3]*expansion)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1]//stride, inputs.shape[2]//stride, inputs.shape[3]*expansion))
        if debug:
            print(prefix + 'depthwise_BN',start_index,end_index)
    else:
        x_mask = np.ones(shape=((inputs.shape[1]//stride, inputs.shape[2]//stride, inputs.shape[3]*expansion)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None, name=prefix + 'project')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (inputs.shape[1]//stride * inputs.shape[2]//stride * pointwise_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1]//stride, inputs.shape[2]//stride, pointwise_filters))
        if debug:
            print(prefix + 'project',start_index,end_index)
    else:
        x_mask = np.ones(shape=((inputs.shape[1]//stride, inputs.shape[2]//stride, pointwise_filters)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)
    #################
    if lambda_mask is not None:
        start_index = end_index
        end_index = start_index + (inputs.shape[1]//stride * inputs.shape[2]//stride * pointwise_filters)
        x_mask  = np.reshape(lambda_mask[start_index:end_index], (inputs.shape[1]//stride, inputs.shape[2]//stride, pointwise_filters))
        if debug:
            print(prefix + 'project_BN',start_index,end_index)
    else:
        x_mask = np.ones(shape=((inputs.shape[1]//stride, inputs.shape[2]//stride, pointwise_filters)))

    x_mask  = backend.variable(x_mask)
    x = Lambda(lambda z: z * x_mask)(x)
    ####################

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def activation_to_magnitude(coordinates):
    """
    Magnitude is proportional to the sum of animate and inanimate activations
    
    dot([x,y],[1,1]) / norm([1,1])
    """
    magnitude = np.sum(coordinates,axis=1)/np.sqrt(2)
    return magnitude 


def activation_to_selectivity(coordinates):
    """
    Selectivity is proportional to Animate - Inanimate activations
    
    dot([x,y],[-1,1]) / norm([-1,1])
    """
    selectivity = (coordinates[:,0] - coordinates[:,1])/np.sqrt(2)
    return selectivity


def grid_space(x,y,y_partitions=28,x_partitions=28,symmetric=False):
    """
    Takes in set of coordinates in 2D space and returns geopandas.GeoDataFrame
    where each entry represents a single cell in the grid space with an equal number
    of units. Symmetric grids don't necessarily have the same number of units per cell.
    
    Parameters
    ----------
        x (1D list) : List of x coordinates
        y (1D list) : List of y coordinates
        y_partitions: Number of partitions in Y direction, should be even number so that
                      cells can be symmetrical around zero
        x_partitions: Number of partitions in X direction, should be even number so that
                      cells can be symmetrical around zero
        symmetric (bool): Whether or not to make it symmetric around zero
                      
    Returns
    -------
        geopandas.GeoDataFrame : one entry per cell in grid starting at bottom left and going right
        
    """
    if symmetric:
        y_neg_sorted = np.sort(y[y<0])
        y_pos_sorted = np.sort(y[y>0])

        # First half of bounds come from negative region, second from positive
        y_bounds = ([y_neg_sorted[int(((2*i)/y_partitions)*len(y_neg_sorted))] 
                         for i in range(int(y_partitions/2))] + [0] +
                    [y_pos_sorted[int(((2*i)/y_partitions)*len(y_pos_sorted))] 
                         for i in range(1,int(y_partitions/2))] + [y_pos_sorted[-1]])
    else:
        y_sorted = np.sort(y)
        y_bounds = ([y_sorted[math.floor((i/y_partitions)*len(y_sorted))] 
                         for i in range(y_partitions)] + [y_sorted[-1]])

    grid_cells = []
    
    for i,y_lower_bound in enumerate(y_bounds[:-1]):
        y_upper_bound = y_bounds[i+1]
        
        if symmetric:
            # Only look at x coordinates which fall within vertical (y direction) strip of interest
            x_neg_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound) & (x < 0)])
            x_pos_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound) & (x > 0)])

            # First half of bounds come from negative region, second from positive
            x_bounds = ([x_neg_sorted[int(((2*k)/x_partitions)*len(x_neg_sorted))] 
                             for k in range(int(x_partitions/2))] + [0] +
                        [x_pos_sorted[int(((2*k)/x_partitions)*len(x_pos_sorted))] 
                             for k in range(1,int(x_partitions/2))] + [x_pos_sorted[-1]])
        else:
            x_sorted = np.sort(x[(y > y_lower_bound) & (y < y_upper_bound)])
            x_bounds = ([x_sorted[int((k/x_partitions)*len(x_sorted))] 
                             for k in range(x_partitions)] + [x_sorted[-1]])
        
        # Add bounds to list
        for j,x_lower_bound in enumerate(x_bounds[:-1]):
            x_upper_bound = x_bounds[j+1]
            # grid_cells.append(shapely.geometry.box(x_lower_bound, y_lower_bound, 
            #                                        x_upper_bound, y_upper_bound))
            grid_cells.append([x_lower_bound, y_lower_bound, 
                               x_upper_bound, y_upper_bound])
    
    # I don't know what this CRS projection is...
    # crs = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
    # return geopandas.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    return grid_cells


#Code snippet needed to read activation values from each layer of the pre-trained artificial neural networks
def get_activations(model, layer, X_batch):
    #keras.backend.function(inputs, outputs, updates=None)
    # print('A ',model.layers[0].input)
    # print('B ',keras.backend.constant(keras.backend.learning_phase()))
    # print('C ',model.layers[layer].output)
    get_activations = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], 
                                             [model.layers[layer].output,])
    # get_activations = keras.backend.function([model.layers[0].input,
    #                                           tf.keras.backend.placeholder(shape=(1,))], 
    #                                          [model.layers[layer].output,])
    #The learning phase flag is a bool tensor (0 = test, 1 = train)
    activations = get_activations([X_batch,0])
    return activations
    
    
def top5accuracy(true, predicted):
    """
    Function to predict the top 5 accuracy
    """
    assert len(true) == len(predicted)
    result = []
    flag  = 0
    for i in range(len(true)):
        flag  = 0
        temp = true[i]
        for j in predicted[i][0:5]:
            if j == temp:
                flag = 1
                break
        if flag == 1:
            result.append(1)
        else:
            result.append(0)
    counter = 0.
    for i in result:
        if i == 1:
            counter += 1.
    error = 1.0 - counter/float(len(result))
    #print len(np.where(np.asarray(result) == 1)[0])
    return len(np.where(np.asarray(result) == 1)[0]), error


def get_ranks(true, predicted):
    assert len(true) == len(predicted)
    ranks = []
    for i,row in enumerate(predicted):
        ranks.append((np.asarray(row)==true[i]).nonzero()[0].item())
    return ranks


def mean_rank_deficit(original_ranks, predicted_ranks):
    """
    Average number of ranks the correct class dropped in predicted ranks compared with
    the true or original ranks. If ranks improved, we can't say anything about the lesion
    so we just return zero which means that no deficit occured.
    """
    assert len(original_ranks) == len(predicted_ranks)
    diff = np.asarray(predicted_ranks,dtype=float)-np.asarray(original_ranks,dtype=float)
    diff[diff < 0] = 0 # If the model improves we count that as no deficit
    return np.mean(diff)


def collect_image_activations(model,image_path_list,existing_activation_ids,save_path=None):
    """
    Construct a M x N matrix, where M is the number of images and N the number of
    neurons by collecting activations from all the neurons in the network.
    """

    for j in range(len(image_path_list)):
        wid = image_path_list[j].split('/')[-1].split('.')[0].split('_')[-1]
        if wid in existing_activation_ids:
            continue
        
        im_temp = preprocess_image_batch([image_path_list[j]],img_size=(256,256), crop_size=(224,224), color_mode="rgb")
        print(j,image_path_list[j])
        data = np.array([],dtype=np.float32)

        i = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if len(weights) > 0:
                activations = get_activations(model,i,im_temp)
                # print activations[0].shape
                temp = np.mean(activations[0], axis=0).ravel()
                print (layer.name, len(temp),len(data))
                if layer.name != 'probs':
                    data = np.append(data, temp)
            i += 1
    
        # Save as we go
        if save_path != None:
            if not os.path.isdir(os.path.join(save_path)):
                os.makedirs(os.path.join(save_path))
            with open(os.path.join(save_path,wid+'.pkl'), 'wb') as handle:
                pickle.dump([data], handle, protocol=pickle.HIGHEST_PROTOCOL)


def euclidean_rdm(activations):
    """
    Calculates a euclidean distance RDM between two M x N activation matrices,
    where M is the number of inputs/examples and N is the number of neuron activations.
    """
    rdm = np.zeros((len(activations),len(activations)))
    for i in range(len(activations)):
        for j in range(len(activations)):
            rdm[i][j] = distance.euclidean(activations[i],activations[j])

    return rdm

# def get_unit_in_cells_dict(cell,selectivity,magnitude):
#     units_in_cells = {}
#     for cell_index in range(len(cell)):
#         units_in_cells[cell_index] = []
#         for unit_index in range(len(magnitude)):
#             # grid_cells.append([x_lower_bound, y_lower_bound, 
#             #                    x_upper_bound, y_upper_bound])
#             if (magnitude[unit_index] >= cell[cell_index][0] and
#                 selectivity[unit_index] >= cell[cell_index][1] and
#                 magnitude[unit_index] <= cell[cell_index][2] and
#                 selectivity[unit_index] <= cell[cell_index][3]):
#                 units_in_cells[cell_index].append(unit_index)
#         print (len(units_in_cells[cell_index]))

#     return units_in_cells

def main():

    def id_to_words(id_):
        return synsets[corr_inv[id_] - 1][2][0]


    def pprint_output(out, n_max_synsets=10):
        wids = []
        best_ids = out.argsort()[::-1][:n_max_synsets]
        for u in best_ids:
            wids.append(str(synsets[corr_inv[u] - 1][1][0]))
        #print('%.2f' % round(100 * out[u], 2) + ' : ' + id_to_words(u)+' '+ str(synsets[corr_inv[u] - 1][1][0]))
        return wids

    model_name = 'mobilenet'
    data_path = '/home/snag-lab/Documents/rsa_tmlr_2023/data/'
    experiment_path = '/home/snag-lab/Documents/rsa_tmlr_2023/data/mobilenet/experiments/'
    # data_path = '../data/'
    
    classes = ['schooner','brain_coral','junco_bird','snail','grey_whale','siberian_husky','electric_fan','bookcase','fountain_pen','toaster']
    class_wids = ['n04147183','n01917289','n01534433','n01944390','n02066245','n02110185','n03271574','n02870880','n03388183','n04442312']
    layer_names = ['Conv1', 'bn_Conv1', 'expanded_conv_depthwise', 'expanded_conv_depthwise_BN', 
                    'expanded_conv_project', 'expanded_conv_project_BN', 'block_1_expand', 'block_1_expand_BN', 
                    'block_1_depthwise', 'block_1_depthwise_BN', 'block_1_project', 'block_1_project_BN', 
                    'block_2_expand', 'block_2_expand_BN', 'block_2_depthwise', 'block_2_depthwise_BN', 
                    'block_2_project', 'block_2_project_BN', 'block_3_expand', 'block_3_expand_BN', 
                    'block_3_depthwise', 'block_3_depthwise_BN', 'block_3_project', 'block_3_project_BN', 
                    'block_4_expand', 'block_4_expand_BN', 'block_4_depthwise', 'block_4_depthwise_BN', 
                    'block_4_project', 'block_4_project_BN', 'block_5_expand', 'block_5_expand_BN', 
                    'block_5_depthwise', 'block_5_depthwise_BN', 'block_5_project', 'block_5_project_BN', 
                    'block_6_expand', 'block_6_expand_BN', 'block_6_depthwise', 'block_6_depthwise_BN', 
                    'block_6_project', 'block_6_project_BN', 'block_7_expand', 'block_7_expand_BN', 
                    'block_7_depthwise', 'block_7_depthwise_BN', 'block_7_project', 'block_7_project_BN', 
                    'block_8_expand', 'block_8_expand_BN', 'block_8_depthwise', 'block_8_depthwise_BN', 
                    'block_8_project', 'block_8_project_BN', 'block_9_expand', 'block_9_expand_BN', 
                    'block_9_depthwise', 'block_9_depthwise_BN', 'block_9_project', 'block_9_project_BN', 
                    'block_10_expand', 'block_10_expand_BN', 'block_10_depthwise', 'block_10_depthwise_BN', 
                    'block_10_project', 'block_10_project_BN', 'block_11_expand', 'block_11_expand_BN', 
                    'block_11_depthwise', 'block_11_depthwise_BN', 'block_11_project', 'block_11_project_BN', 
                    'block_12_expand', 'block_12_expand_BN', 'block_12_depthwise', 'block_12_depthwise_BN', 
                    'block_12_project', 'block_12_project_BN', 'block_13_expand', 'block_13_expand_BN', 
                    'block_13_depthwise', 'block_13_depthwise_BN', 'block_13_project', 'block_13_project_BN', 
                    'block_14_expand', 'block_14_expand_BN', 'block_14_depthwise', 'block_14_depthwise_BN', 
                    'block_14_project', 'block_14_project_BN', 'block_15_expand', 'block_15_expand_BN', 
                    'block_15_depthwise', 'block_15_depthwise_BN', 'block_15_project', 'block_15_project_BN', 
                    'block_16_expand', 'block_16_expand_BN', 'block_16_depthwise', 'block_16_depthwise_BN', 
                    'block_16_project', 'block_16_project_BN', 'Conv_1', 'Conv_1_bn']

    layer_indexes = [0, 200704, 401408, 602112, 802816, 903168, 1003520, 1605632, 2207744, 2358272, 2508800, 
    2533888, 2558976, 2709504, 2860032, 3010560, 3161088, 3186176, 3211264, 3361792, 3512320, 3549952, 3587584, 
    3600128, 3612672, 3687936, 3763200, 3838464, 3913728, 3926272, 3938816, 4014080, 4089344, 4164608, 4239872, 
    4252416, 4264960, 4340224, 4415488, 4434304, 4453120, 4457824, 4462528, 4490752, 4518976, 4547200, 4575424, 
    4580128, 4584832, 4613056, 4641280, 4669504, 4697728, 4702432, 4707136, 4735360, 4763584, 4791808, 4820032, 
    4824736, 4829440, 4857664, 4885888, 4914112, 4942336, 4948608, 4954880, 4992512, 5030144, 5067776, 5105408, 
    5111680, 5117952, 5155584, 5193216, 5230848, 5268480, 5274752, 5281024, 5318656, 5356288, 5365696, 5375104,
    5377848, 5380592, 5397056, 5413520, 5429984, 5446448, 5449192, 5451936, 5468400, 5484864, 5501328, 5517792, 
    5520536, 5523280, 5539744, 5556208, 5572672, 5589136, 5594624, 5600112, 5662832, 5725552]

    # # Using command line to process classes selectively
    # args = sys.argv[1:]
    # if len(args)!=0 and args[0] == '-classes':
    #     classes_to_process = args[1:]
    # else:
    #     classes_to_process = classes

    # if len(args)!=0 and '-layer_start' in args:
    #     layer_start = args[args.index('-layer_start')+1]
    # else:
    #     layer_start = layers[0]

    #Load the details of all the 1000 classes and the function to convert the synset id to words
    meta_clsloc_file = data_path+'meta_clsloc.mat'
    synsets = loadmat(meta_clsloc_file)['synsets'][0]
    synsets_imagenet_sorted = sorted([(int(s[0]), str(s[1][0])) for s in synsets[:1000]],key=lambda v: v[1])
    corr = {}
    for j in range(1000):
        corr[synsets_imagenet_sorted[j][0]] = j

    corr_inv = {}
    for j in range(1, 1001):
        corr_inv[corr[j]] = j

    #Code snippet to load the ground truth labels to measure the performance
    truth = {}
    with open(data_path+'ILSVRC2014_clsloc_validation_ground_truth.txt') as f:
        line_num = 1
        for line in f.readlines():
            ind_ = int(line)
            temp  = None
            for i in synsets_imagenet_sorted:
                if i[0] == ind_:
                    temp = i
            #print ind_,temp
            if temp != None:
                truth[line_num] = temp
            else:
                print('##########', ind_)
                pass
            line_num += 1
    
    # Get list of all animate and inanimate images
    im_valid_test = glob.glob(data_path+'images/*')
    im_valid_test = np.asarray(im_valid_test)

    # Make list of wids
    true_valid_wids = []
    for i in im_valid_test:
        temp1 = i.split('/')[-1]
        temp = temp1.split('.')[0].split('_')[-1]
        true_valid_wids.append(truth[int(temp)][1])
    true_valid_wids = np.asarray(true_valid_wids)

    print ('loading activations')
    model = MobileNetV2(input_shape=None, alpha=0.35,include_top=True, weights="imagenet",input_tensor=None,
                        pooling=None, classes=1000, classifier_activation="softmax", 
                        lambda_mask = np.ones(shape=((5725552,))))

    # If activations for all neurons for all images exist, grab those
    # Average them across images in each category and save them
    # If activations for all neurons for all images exist, grab those
    # Average them across images in each category and save them
    if os.path.isdir(os.path.join(data_path,model_name,'activations')):
        existing_activations = os.listdir(os.path.join(data_path,model_name,'activations'))
        existing_activation_ids = [split('\.',di)[0] for di in existing_activations]
    
    if len(existing_activations)<len(im_valid_test):
        collect_image_activations(model,im_valid_test,existing_activation_ids,
                                    os.path.join(data_path,model_name,'activations'))

    # unit_activations = []
    # for im in im_valid_test:
    #     wid = im.split('/')[-1].split('.')[0].split('_')[-1]
    #     with open(os.path.join(data_path,model_name,'activations',wid+'.pkl'), 'rb') as f:
    #         unit_activations.append(pickle.load(f)[0])

    # unit_activations = np.row_stack(unit_activations)
    # print(unit_activations.shape)

    # print ('calculating baseline ranks')
    # im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
    # out = model.predict(im_temp,batch_size=64)

    # predicted_valid_wids = []
    # for i in range(len(im_valid_test)):
    #     predicted_valid_wids.append(pprint_output(out[i],1000))
    # predicted_valid_wids = np.asarray(predicted_valid_wids)

    # # Count errors and save baseline ranks
    # count, error  = top5accuracy(true_valid_wids, predicted_valid_wids)
    # baseline_ranks  = np.asarray(get_ranks(true_valid_wids,predicted_valid_wids))

    # print (baseline_ranks.shape)
    # print('baseline '+str(count)+' '+str(len(true_valid_wids))+' '+str(error)+' '+str(1-error))

    # keras.backend.clear_session()
    # gc.collect()
    # del model

    # # For each class
    #     # Calculate Selectivity and Magnitude
    #     # Grid the space
    #     # For each cell
    #         # Calculate SRD
    #         # Calculate SRS
    # # Estimate Distribution of 
    #     # SRD
    #     # SRS

    # for class_idx,c in enumerate(classes):
    #     layer_start_seen = False

    #     if c not in classes_to_process:
    #         continue
        
    #     print ('calculating activations')
    #     class_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]==class_wids[class_idx]]
    #     other_indexes = [idx for idx in range(len(true_valid_wids)) if true_valid_wids[idx]!=class_wids[class_idx]]
    #     class_activations = np.mean(unit_activations[class_indexes],axis=0)
    #     other_activations = np.mean(unit_activations[other_indexes],axis=0)

    #     X = np.column_stack((class_activations,other_activations))

    #     # Labels correspond to class indexes
    #     y = np.asarray([1 if i in class_indexes else 0 for i in range(len(unit_activations))])

    #     for layer_idx,layer in enumerate(layer_names):
    #         if not layer_start_seen:
    #             if layer == layer_start:
    #                 layer_start_seen = True
    #             else:
    #                 continue
    #         # Magnitude and selectivity on a per layer basis
    #         magnitude = activation_to_magnitude(X[layer_indexes[layer_idx]:layer_indexes[layer_idx+1]])
    #         selectivity = activation_to_selectivity(X[layer_indexes[layer_idx]:layer_indexes[layer_idx+1]])

    #         print ('generating grid')
    #         x_partitions,y_partitions = 4,4
    #         cell = grid_space(magnitude,selectivity,x_partitions=x_partitions,y_partitions=y_partitions)

    #         with open(os.path.join(experiment_path,model_name,'grid_specifications_'+c+'_'+layer+'.pkl'), 'wb') as handle:
    #             pickle.dump([cell], handle, protocol=pickle.HIGHEST_PROTOCOL)

    #         # Don't have access to geopandas in python 2.7, so we're brute forcing unit assignments to cells.
    #         units_in_cells = dict()
    #         for cell_index in range(len(cell)):
    #             units_in_cells[cell_index] = []
    #             for unit_index in range(len(magnitude)):
    #                 network_unit_index = unit_index + layer_indexes[layer_idx] # Offset to get back the overall network index
    #                 if (magnitude[unit_index] >= cell[cell_index][0] and
    #                     selectivity[unit_index] >= cell[cell_index][1] and
    #                     magnitude[unit_index] <= cell[cell_index][2] and
    #                     selectivity[unit_index] <= cell[cell_index][3]):
    #                     units_in_cells[cell_index].append(network_unit_index)
    #             print (len(units_in_cells[cell_index]),
    #                     min(units_in_cells[cell_index]),
    #                     max(units_in_cells[cell_index]))

    #         with open(os.path.join(experiment_path,model_name,'units_in_cells_'+c+'_'+layer+'.pkl'), 'wb') as handle:
    #             pickle.dump([units_in_cells], handle, protocol=pickle.HIGHEST_PROTOCOL)

    #         # for each cell in grid
    #         print ('calculating rank deficits for each cell')
    #         cell_srd = {}
    #         for bbx in range(len(cell)):
    #             start = time.time()
                
    #             # Query indices of units in that cell, create mask and set activations to zero
    #             loc_new = units_in_cells[bbx]
    #             lambda_mask = np.ones(shape=((5725552,)))

    #             lambda_mask[loc_new] = 0.
    #             print('Cell: ', bbx, ' Units: ', len(loc_new))
                
    #             # Skip this cell if no units lie within it
    #             if len(loc_new) == 0.:
    #                 cell_srd[bbx] = [0,0,0]
    #                 continue

    #             model = MobileNetV2(input_shape=None, alpha=0.35,include_top=True, weights="imagenet",input_tensor=None,
    #                     pooling=None, classes=1000, classifier_activation="softmax", 
    #                     lambda_mask=lambda_mask)                         
                
    #             im_temp = preprocess_image_batch(im_valid_test,img_size=(256,256), crop_size=(224,224), color_mode="rgb")
    #             out = model.predict(im_temp,batch_size=64)

    #             predicted_valid_wids = []
    #             for i in range(len(im_valid_test)):
    #                 predicted_valid_wids.append(pprint_output(out[i],1000))
    #             predicted_valid_wids = np.asarray(predicted_valid_wids)

    #             # calculate ranks
    #             count, error  = top5accuracy(true_valid_wids[class_indexes], predicted_valid_wids[class_indexes])
    #             class_ranks  = get_ranks(true_valid_wids[class_indexes],
    #                                     predicted_valid_wids[class_indexes])
    #             other_ranks  = get_ranks(true_valid_wids[other_indexes],
    #                                     predicted_valid_wids[other_indexes])
    #             class_mrd = mean_rank_deficit(baseline_ranks[class_indexes],class_ranks)
    #             other_mrd = mean_rank_deficit(baseline_ranks[other_indexes],other_ranks)

    #             print(class_mrd,other_mrd)
    #             print(c+' '+str(count)+' '+str(len(class_indexes))+' '+str(error)+' '+str(1-error))
            
    #             srd_score = class_mrd - other_mrd
    #             cell_srd[bbx] = [srd_score, class_mrd, other_mrd]

    #             keras.backend.clear_session()
    #             gc.collect()
    #             del model
    #             print("time : ", time.time()-start)

    #         # Dump SRD
    #         with open(os.path.join(experiment_path,model_name,'srd_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
    #             pickle.dump([cell_srd], handle, protocol=pickle.HIGHEST_PROTOCOL)

    #         # Create class template RDM
    #         class_template_RDM = np.ones((len(im_valid_test),len(im_valid_test)))
    #         for row in range(len(im_valid_test)):
    #             for column in range(len(im_valid_test)):
    #                 if ((row in class_indexes) and (column in class_indexes)) or (row == column):
    #                     class_template_RDM[row][column] = 0

    #         # Semantic scores for each cell calculation
    #         print ('calculating semantic score for each cell')
    #         srs_result = {}
    #         cell_probe = {}
    #         for bbx in range(len(cell)):
    #             start = time.time() 
    #             loc_new = units_in_cells[bbx]
    #             print('Cell: ', bbx,'Units: ',len(loc_new))

    #             if len(loc_new) == 0.:
    #                 srs_result[bbx] = [0,0,0,0]
    #                 continue

    #             # All images, only units in cell
    #             act = unit_activations[:,loc_new]
    #             cell_RDM_pearson = 1 - np.corrcoef(act)
    #             cell_RDM_euclidean = euclidean_rdm(act)    
    #             srs_result[bbx]    = [stats.kendalltau(cell_RDM_pearson,class_template_RDM)[0],
    #                                 stats.kendalltau(cell_RDM_euclidean,class_template_RDM)[0],
    #                                 stats.spearmanr(cell_RDM_pearson,class_template_RDM,axis=None)[0],
    #                                 stats.spearmanr(cell_RDM_euclidean,class_template_RDM,axis=None)[0]]
    #             print (srs_result[bbx])
    #             print("time : ", time.time()-start)

    #             # Fit classifier to 200 groups of 100 random neurons and record losses
    #             losses = []
    #             start = time.time()
    #             for _ in range(200):
    #                 clf = LogisticRegression(penalty='none',random_state=0)
    #                 random_indexes = np.random.choice(np.arange(act.shape[1]),
    #                                                     size=min(act.shape[1],100),
    #                                                     replace=False)
    #                 clf.fit(act[:,random_indexes],y)
    #                 losses.append(log_loss(y,clf.predict(act[:,random_indexes])))
                
    #             cell_probe[bbx] = [np.mean(losses)]
    #             print(np.mean(losses),time.time()-start)

    #         with open(os.path.join(experiment_path,model_name,'rsa_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
    #             pickle.dump([srs_result], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #         with open(os.path.join(experiment_path,model_name,'linear_probe_grid_4x4_'+c+'_'+layer+'.pkl'), 'wb') as handle:
    #                 pickle.dump([cell_probe], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()