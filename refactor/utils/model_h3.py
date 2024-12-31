# model_h3.py

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, Activation, Flatten, 
    MaxPooling2D, LayerNormalization, AveragePooling2D, add
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from keras.initializers import glorot_uniform
import keras

###############################################################################
'''
Functions
'''
###############################################################################

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 160:
        lr *= 0.5e-3
    elif epoch > 120:
        lr *= 1e-3
    elif epoch > 80:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def Conv_bn_relu(infor, **conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    conv = Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer)(infor)

    norm = LayerNormalization(axis=CHANNEL_AXIS)(conv)
    out = Activation("relu")(norm)
    return out


def Bn_relu_conv(infor, **conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    norm = LayerNormalization(axis=CHANNEL_AXIS)(infor)
    activation = Activation("relu")(norm)

    out = Conv2D(filters=filters, kernel_size=kernel_size,
                 strides=strides, padding=padding,
                 kernel_initializer=kernel_initializer,
                 kernel_regularizer=kernel_regularizer)(activation)
    return out


def basic_block(BlockIn, filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    if is_first_block_of_first_layer:
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=init_strides,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4))(BlockIn)
    else:
        conv1 = Bn_relu_conv(infor=BlockIn, filters=filters, kernel_size=(3, 3), strides=init_strides)

    residual = Bn_relu_conv(infor=conv1, filters=filters, kernel_size=(3, 3))

    input_shape = K.int_shape(BlockIn)
    residual_shape = K.int_shape(residual)

    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(BlockIn)
    else:
        shortcut = BlockIn

    return add([shortcut, residual])


def buildmodel(input_shape, num_outputs, Regress_Flag):
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS

    if tf.keras.backend.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3

    if len(input_shape) != 3:
        raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

    # Permute dimension order if necessary
    if K.image_data_format() == 'channels_last':
        # input_shape is (channels, rows, cols) => convert to (rows, cols, channels)
        input_shape = (input_shape[1], input_shape[2], input_shape[0])

    inputs = Input(shape=input_shape, name='main_input')
    conv1 = Conv_bn_relu(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2))
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
                BlockIn=block,
                filters=filters,
                init_strides=init_strides,
                is_first_block_of_first_layer=(is_first_layer and j == 0)
            )
        filters *= 2

    norm = LayerNormalization(axis=CHANNEL_AXIS)(block)
    block = Activation("relu")(norm)

    block_shape = K.int_shape(block)
    pool2_out = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1),
                                 name="conv_final")(block)

    flatten1 = Flatten(name="Flatten")(pool2_out)
    flatten1 = Dense(128, activation='relu', name="Flatten2")(flatten1)

    if not Regress_Flag:
        # Classification
        dense = Dense(
            units=num_outputs, 
            kernel_initializer="he_normal",
            activation="softmax", 
            name="Dense_layer"
        )(flatten1)
    else:
        # Regression
        dense = Dense(
            units=1, 
            activation='linear',
            kernel_initializer=glorot_uniform(seed=0)
        )(flatten1)

    model = Model(inputs=inputs, outputs=dense)
    return model


def learn_model(oldmodel, nb_classes, Transfer_Type, summary=False):
    """
    Transfer from the old model to a new classification or regression head.
    By default, we do call `model.compile()`. If TFF must see an uncompiled 
    model, you can remove or conditionally skip these compile calls.
    """
    base_model = oldmodel

    if Transfer_Type == 'Classification':
        # Grab up to conv_final
        intermediate_layer_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("conv_final").output
        )
    else:  # Regression
        # We assume your old classification layer is named "Dense_Classification"
        # so we can start from that layer or flatten?
        intermediate_layer_model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("Dense_Classification").output
        )

    x = intermediate_layer_model(base_model.input)
    # If `nb_classes > 1` for classification, we might do a global average pooling
    # For single-output regression, you could skip it, or do GAP if you wish:
    if nb_classes > 1 and Transfer_Type == 'Classification':
        x = keras.layers.GlobalAveragePooling2D()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    if Transfer_Type == 'Classification':
        dense = Dense(
            units=nb_classes, 
            kernel_initializer="he_normal",
            activation="softmax", 
            name="Dense_Classification"
        )(x)
    else:  # Regression
        dense = Dense(
            units=1, 
            activation='linear',
            kernel_initializer=glorot_uniform(seed=0),
            name="Dense_Regression"
        )(x)

    model = Model(inputs=base_model.input, outputs=dense)

    if summary:
        model.summary()

    # If you want TFF to see an uncompiled model, remove or guard this:
    if Transfer_Type == 'Classification':
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return model
