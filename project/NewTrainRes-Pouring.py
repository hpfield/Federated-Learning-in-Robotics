# -*- coding: utf-8 -*-
###############################################################################
'''
Import library
'''
###############################################################################

from __future__ import print_function

import keras
from keras.layers.convolutional import ( Conv2D, MaxPooling2D, AveragePooling2D)
from keras.layers import (    Input,    Activation,    Dense,    Flatten)
from keras.layers import add
#from keras.layers.normalization import BatchNormalization
from keras.layers import LayerNormalization

from keras.regularizers import l2
from keras import backend as K
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils import np_utils

import os
import numpy as np
import Config

from sklearn.model_selection import train_test_split

###############################################################################
'''
Functions
'''
###############################################################################

from keras.initializers import glorot_uniform

def lr_schedule(epoch):
    '''
    epoch: number of epochs for model training
    lr: learning rate
    ''' 
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
    '''
    Build conv -> BN -> relu block
    '''
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


#Reference: http://arxiv.org/pdf/1603.05027v2.pdf
def Bn_relu_conv(infor,**conv_params):
    '''
    Build a BN -> relu -> conv block.
    '''
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
    '''
    Basic 3 X 3 convolution blocks
    '''

    if is_first_block_of_first_layer:
        conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                       strides=init_strides,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=l2(1e-4))(BlockIn)
    else:
        conv1 = Bn_relu_conv(infor = BlockIn,filters=filters, kernel_size=(3, 3),
                              strides=init_strides)

    residual = Bn_relu_conv(infor = conv1,filters=filters, kernel_size=(3, 3))


    input_shape = K.int_shape(BlockIn)
    residual_shape = K.int_shape(residual)
    
    # stride should be set properly and match  (width, height) of residual
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
    
    # Adds a shortcut between input and residual block
    return add([shortcut, residual])


def buildmodel(input_shape, num_outputs, Regress_Flag):
        '''

        input_shape: (nb_channels, nb_rows, nb_cols)
        num_outputs:  number of outputs at final softmax layer
        Regress_Flag: classify or regress

        '''

        global ROW_AXIS
        global COL_AXIS
        global CHANNEL_AXIS
        
        if K.image_data_format() == 'channels_last':
        
            ROW_AXIS = 1; COL_AXIS = 2;  CHANNEL_AXIS = 3
        else:
            CHANNEL_AXIS = 1; ROW_AXIS = 2; COL_AXIS = 3        
        
        
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
        #if K.image_dim_ordering() == 'tf':

            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape, name='main_input')
        conv1 = Conv_bn_relu(input, filters=64, kernel_size=(7, 7), strides=(2, 2))
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name = "conv_pool1")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate([2,2,2,2]):
            '''
            residual block with repeating bottleneck blocks
            '''

            is_first_layer=(i == 0)
            for i in range(r):
                init_strides = (1, 1)
                if i == 0 and not is_first_layer:
                    init_strides = (2, 2)
                block = basic_block(BlockIn = block, filters=filters, init_strides=init_strides,
                                       is_first_block_of_first_layer=(is_first_layer and i == 0))
                
            filters *= 2

        # Last activation
    
        norm = LayerNormalization(axis=CHANNEL_AXIS)(block)
        block = Activation("relu")(norm)

        # classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1), name = "conv_final")(block)
        
        #flatten1 = keras.layers.GlobalAveragePooling2D(name = "GAP")(pool2)
        #out = keras.layers.Dense(num_outputs,activation='softmax')(pooled)

        flatten1 = Flatten( name = "Flatten")(pool2)

        flatten1 = Dense(128, activation='relu',name = "Flatten2")(flatten1)

        if Regress_Flag == False:
            
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax", name = "Dense_layer")(flatten1)
        else:
            dense = Dense(units=1, activation='linear', kernel_initializer=glorot_uniform(seed=0))(flatten1)
            
            #dense = Dense(units=1, kernel_initializer="he_normal", activation="linear", name = "Dense_layer")(flatten1)
  
        model = Model(inputs=input, outputs=dense)
        return model


def learn_model(oldmodel,nb_classes,Transfer_Type,summary=False): 

    base_model = oldmodel
    
    if Transfer_Type == 'Classification':
        intermediate_layer_model = Model(inputs=base_model.input,outputs=base_model.get_layer("conv_final").output)

    if Transfer_Type == 'Regression':
        #intermediate_layer_model = Model(inputs=base_model.input,outputs=base_model.get_layer("Flatten1").output)
        intermediate_layer_model = Model(inputs=base_model.input,outputs=base_model.get_layer("Dense_Classification").output)
    
    x = intermediate_layer_model(base_model.input)
    if nb_classes >1:
        x = keras.layers.GlobalAveragePooling2D()(x)# 添加全局平均池化层
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    
    
    if Transfer_Type == 'Classification':
        dense = Dense(units=nb_classes, kernel_initializer="he_normal",
                  activation="softmax", name = "Dense_Classification")(x)
    if Transfer_Type == 'Regression':
        dense = Dense(units=1, activation='linear', name = "Dense_Regression", kernel_initializer=glorot_uniform(seed=0))(x)

    model = Model(inputs=base_model.input, outputs=dense)
    
    
	# show summary if specified 
    if summary==True : 
	    model.summary()

    if Transfer_Type == 'Classification':
    	# choose the optimizer 
        #optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    if Transfer_Type == 'Regression':
        model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics=['mse'])

    return model
 

 
###############################################################################

'''
Load Data
'''

###############################################################################   
 
NPZ_Name = Config.Folder_Name + '_' + Config.Type +  '_' + str(Config.img_rows) +  '_overall_database.npz'
Database_Used = np.load(NPZ_Name)
X_train = Database_Used['X_train']
Y_train_State = Database_Used['Y_train_State']
Y_train_Phase = Database_Used['Y_train_Context']
Y_train_Regress=Database_Used['Y_train_Regress']


lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)


print('Using real-time data augmentation.')

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    #rotation_range= 5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
    #shear_range=0.01,
    #zoom_range=0.05,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

#! Setup save directory
save_dir = os.path.join(os.getcwd(), 'Log')

############################################################################### 

'''
Model training for the 1st hierarchy
'''

############################################################################### 

'''
Prepare to train model H1
'''
nb_classes = 4 
Model_Type = 'Phase'

print('Classify Phase!')
Y_train = Y_train_Phase
Y_train = np_utils.to_categorical(Y_train, nb_classes)

X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=100)

print('Train Size:' , np.shape(Y_train),'Test Size:' , np.shape(Y_test))


Logger_Name = Config.Folder_Name + '_' + Config.Type +   '_' + str(Config.img_rows) +  '_' + Model_Type + '_results.csv'    

Logger_Name = os.path.join(save_dir, Logger_Name)

csv_logger = CSVLogger(Logger_Name)

model = buildmodel((Config.img_channels, Config.img_rows, Config.img_cols), nb_classes,False)
#model.summary()

###############################################################################
# no use
Version = 1
model_type = 'ResNetv%d' % (Version)
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'Log')
model_name = 'Robot_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
print('file path for model:', filepath)

###############################################################################
'''
Construct model H1
'''
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             period=10,
                             save_best_only=False)    
callbacks = [checkpoint, lr_reducer, lr_scheduler]    
###############################################################################    
'''
Train model H1
'''
# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=Config.batch_size),
                        steps_per_epoch=X_train.shape[0] // Config.batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=Config.nb_epoch, verbose=1, 
                        callbacks=[lr_reducer, early_stopper, csv_logger])  
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=Config.batch_size),
                        steps_per_epoch=X_train.shape[0] // Config.batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=Config.nb_epoch, verbose=1, 
                        callbacks=[lr_reducer, early_stopper, csv_logger])    
# class_weight=[0.2,0.1,0.5,0.2],
#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=Config.batch_size))    
  
  

Model_Name = Config.Folder_Name + Config.Type +  '_' + str(Config.img_rows) +  '_' + Model_Type + '_model.h5'
    
Model_Name = os.path.join(save_dir, Model_Name)

model.save(Model_Name)
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

############################################################################### 

'''
Model training for the 2nd hierarchy
'''

############################################################################### 

'''
Prepare to train model H2
'''

Model_Type = 'State'

print('Classify State!')
Transfer_Type = 'Classification'
NPZ_Name = Config.Folder_Name + '_' + Config.Type +  '_' + str(Config.img_rows) +  '_overall_database.npz'
Database_Used = np.load(NPZ_Name)
X_train = Database_Used['X_train']
Y_train = Y_train_State
Y_train = np_utils.to_categorical(Y_train, Config.nb_classes)

X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=100)

print('Train Size:' , np.shape(Y_train),'Test Size:' , np.shape(Y_test))


############################################################################### 
'''
Construct model H2
'''

Old_ModelName =  Config.Folder_Name + Config.Type +  '_' + str(Config.img_rows) +  '_' + 'Phase' + '_model.h5'
Old_ModelName = os.path.join(save_dir, Old_ModelName)
oldmodel = keras.models.load_model(Old_ModelName)

model = learn_model(oldmodel,Config.nb_classes,Transfer_Type,summary=False)
Logger_Name = Config.Folder_Name + '_' + Config.Type +   '_' + str(Config.img_rows) +  '_' + Model_Type + '_results.csv'
Logger_Name = os.path.join(save_dir, Logger_Name)
csv_logger = CSVLogger(Logger_Name)



model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             period=10,
                             save_best_only=False) 
callbacks = [checkpoint, lr_reducer, lr_scheduler]    


############################################################################### 
'''
Train model H1
'''

    
# Fit the model on the batches generated by datagen.flow().
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=Config.batch_size),
                    steps_per_epoch=X_train.shape[0] // Config.batch_size,
                    validation_data=(X_test, Y_test),
                    epochs=Config.nb_epoch, verbose=1, 
                    callbacks=[lr_reducer, early_stopper, csv_logger])    
    
#from keras.utils import plot_model   
#plot_model(model, to_file='ResNet18_model.png', show_shapes=True) 

Model_Name = Config.Folder_Name + Config.Type +  '_' + str(Config.img_rows) +  '_' + Model_Type + '_model.h5'

Model_Name = os.path.join(save_dir, Model_Name)

model.save(Model_Name)
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1]) 



############################################################################### 

'''
Model training for the 3rd hierarchy
'''

############################################################################### 

'''
Prepare to train model H1
'''

Model_Type = 'Action'
nb_classes=1
print('Regression !')
Transfer_Type = 'Regression'
NPZ_Name = Config.Folder_Name + '_' + Config.Type +  '_' + str(Config.img_rows) +  '_overall_database.npz'
Database_Used = np.load(NPZ_Name)
X_train = Database_Used['X_train']
Y_train = Y_train_Regress


X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.2,random_state=100)

print('Train Size:' , np.shape(Y_train),'Test Size:' , np.shape(Y_test))
Logger_Name = Config.Folder_Name + '_' + Config.Type +   '_' + str(Config.img_rows) +  '_' + Model_Type + '_results.csv'
Logger_Name = os.path.join(save_dir, Logger_Name)
csv_logger = CSVLogger(Logger_Name)


############################################################################### 

'''
Construct model H3
'''


print('Load_Transfer!!!')
Old_ModelName =  Config.Folder_Name + Config.Type +  '_' + str(Config.img_rows) +  '_' + 'State' + '_model.h5'
Old_ModelName = os.path.join(save_dir, Old_ModelName)
oldmodel = keras.models.load_model(Old_ModelName)
model = learn_model(oldmodel,nb_classes,Transfer_Type,summary=False)
Logger_Name = Config.Folder_Name + '_' + Config.Type +   '_' + str(Config.img_rows) +  '_' + Model_Type + '_transfer_resnet.csv'
Logger_Name = os.path.join(save_dir, Logger_Name)




model.compile(loss = 'mean_squared_error',
              optimizer = 'adam', 
              metrics=['mse'])

checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_mse',
                             verbose=1,
                             period=1,
                             save_best_only=False)    
callbacks = [checkpoint, lr_reducer, lr_scheduler]   

############################################################################### 

'''
Train model H3
'''


model.fit_generator(datagen.flow(X_train, Y_train, batch_size=Config.batch_size),
                        steps_per_epoch=X_train.shape[0] // Config.batch_size,
                        validation_data=(X_test, Y_test),
                        epochs=Config.nb_epoch, verbose=1, 
                        callbacks=[lr_reducer, early_stopper, csv_logger])   

Model_Name = Config.Folder_Name + Config.Type +  '_' + str(Config.img_rows) +  '_' + Model_Type + '__model.h5'
 
Model_Name = os.path.join(save_dir, Model_Name)

model.save(Model_Name)
scores = model.evaluate(X_test, Y_test, verbose=1)



