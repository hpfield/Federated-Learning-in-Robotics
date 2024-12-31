# utilities/model_h1.py
import tensorflow as tf
import keras
from keras.layers import (
    Conv2D, MaxPooling2D, AveragePooling2D, Activation, 
    Dense, Flatten, add, LayerNormalization, Input
)
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.initializers import glorot_uniform

# For consistent naming of axes in TFF/TF
ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3

def lr_schedule(epoch):
    """
    Learning rate schedule from the original code (not used in TFF, but left unchanged).
    """
    lr = 1e-3
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    logger.info('Learning rate:', lr)
    return lr


def Conv_bn_relu(x, **conv_params):
    """
    Build conv -> BN -> relu block.
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    conv = Conv2D(
        filters=filters, 
        kernel_size=kernel_size,
        strides=strides, 
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(x)

    norm = LayerNormalization(axis=CHANNEL_AXIS)(conv)
    out = Activation("relu")(norm)
    return out


def Bn_relu_conv(x, **conv_params):
    """
    Build a BN -> relu -> conv block.
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    norm = LayerNormalization(axis=CHANNEL_AXIS)(x)
    activation = Activation("relu")(norm)

    out = Conv2D(
        filters=filters, 
        kernel_size=kernel_size,
        strides=strides, 
        padding=padding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer
    )(activation)
    return out


def basic_block(block_in, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """
    Basic residual block with 3x3 convolutions.
    """
    if is_first_block_of_first_layer:
        conv1 = Conv2D(
            filters=filters, kernel_size=(3, 3),
            strides=init_strides,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4)
        )(block_in)
    else:
        conv1 = Bn_relu_conv(
            x=block_in, filters=filters, kernel_size=(3, 3),
            strides=init_strides
        )

    residual = Bn_relu_conv(x=conv1, filters=filters, kernel_size=(3, 3))

    input_shape = K.int_shape(block_in)
    residual_shape = K.int_shape(residual)

    stride_width  = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))

    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(
            filters=residual_shape[CHANNEL_AXIS],
            kernel_size=(1, 1),
            strides=(stride_width, stride_height),
            padding="valid",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(0.0001)
        )(block_in)
    else:
        shortcut = block_in

    return add([shortcut, residual])


def buildmodel(input_shape, num_outputs, Regress_Flag):
    """
    Build the base ResNet-like model used for H1.
    """
    global ROW_AXIS, COL_AXIS, CHANNEL_AXIS

    if tf.keras.backend.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

    # Input layer
    inp = Input(shape=input_shape, name='main_input')

    # First conv + pool
    conv1 = Conv_bn_relu(inp, filters=64, kernel_size=(7, 7), strides=(2, 2))
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="conv_pool1")(conv1)

    block = pool1
    filters = 64
    for i, r in enumerate([2, 2, 2, 2]):
        is_first_layer = (i == 0)
        for j in range(r):
            init_strides = (1, 1)
            if j == 0 and not is_first_layer:
                init_strides = (2, 2)
            block = basic_block(
                block_in=block, 
                filters=filters, 
                init_strides=init_strides,
                is_first_block_of_first_layer=(is_first_layer and j == 0)
            )
        filters *= 2

    norm = LayerNormalization(axis=CHANNEL_AXIS)(block)
    block = Activation("relu")(norm)

    block_shape = K.int_shape(block)
    pool2_out = AveragePooling2D(
        pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
        strides=(1, 1), 
        name="conv_final"
    )(block)

    flatten1 = Flatten(name="Flatten")(pool2_out)
    flatten1 = Dense(128, activation='relu', name="Flatten2")(flatten1)

    if not Regress_Flag:
        dense = Dense(
            units=num_outputs, 
            kernel_initializer="he_normal",
            activation="softmax", 
            name="Dense_layer"
        )(flatten1)
    else:
        dense = Dense(
            units=1, 
            kernel_initializer=glorot_uniform(seed=0),
            activation="linear", 
            name="Dense_layer"
        )(flatten1)

    model = Model(inputs=inp, outputs=dense)
    return model


def build_h1_model(img_channels, img_rows, img_cols, phase_classes):
    # channels-last shape is (img_rows, img_cols, img_channels)
    return buildmodel(
        (img_rows, img_cols, img_channels), 
        phase_classes, 
        Regress_Flag=False
    )

