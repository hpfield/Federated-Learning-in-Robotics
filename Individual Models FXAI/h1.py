

# %%
from __future__ import print_function

import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import collections
from tensorflow.keras.models import load_model  # Use tensorflow.keras consistently

# From EHIL

from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D)
from tensorflow.keras.layers import (Input, Activation, Dense, Flatten)
from tensorflow.keras.layers import add, LayerNormalization

from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.utils import to_categorical  # Replace np_utils
from tensorflow.keras.utils import plot_model

import os
from sklearn.model_selection import train_test_split

import time
import datetime
from tqdm import tqdm


# %%
print(tf.__version__)

# %% [markdown]
# ## Config

# %%
img_channels = 3
BATCH_SIZE = 20
NUM_CLIENTS = 25
PHASE_CLASSES = 4
EPOCHS = 100
LOGS_DIR = 'H1'
MODEL_NAME = 'h1_25_clients_100_epochs'

##############################################################################


img_rows, img_cols = 256,256
img_rows, img_cols = 50,50

Cup_Type = 'Big'
if Cup_Type == 'Medium':
    nb_classes = 9
if Cup_Type == 'Big':
    nb_classes = 10
if Cup_Type == 'Small':
    nb_classes = 7

# %% [markdown]
# We set up the summary writer to log our model's performance through training

# %%
summary_writer_h1 = tf.summary.create_file_writer('logs/' + LOGS_DIR)

# %% [markdown]
# ## Load Data

# %%
NPZ_Name = 'Data/Videos_Database_20_Robot_WebCam_50_overall_database.npz'
Database_Used = np.load(NPZ_Name)
Sessions = Database_Used['Session']

# %%
# create a list of the unique sessions to become the client_ids
client_ids = np.unique(Sessions)

# %%
# Several steps need to be completed to convert our data to a format suitable for tensorflow-federated operations
# The first is to create tf.data.Dataset objects
def create_tf_dataset(client_ids, Database_used_col, categorical, categories):
  train_datasets = []
  test_datasets = []
  for session in client_ids:
    # find the indices of the current session in the Sessions column of Database_Used
    session_indices = np.where(Sessions == session)[0]

    # get the X_train data for the current session
    session_X = Database_Used['X_train'][session_indices]
    # grab the training data for the necessary hierarchy
    session_Y = Database_Used[Database_used_col][session_indices]
    # if using categorical data, reshape the data for the model into one-hot encoded
    if categorical==True:
      session_Y = tf.keras.utils.to_categorical(session_Y, categories)
    # create train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(session_X, session_Y, test_size=0.2, random_state=100)
    # Make into tf dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    # Add to our list of datasets
    train_datasets.append([session,train_dataset])
    test_datasets.append([session,test_dataset])
  return train_datasets, test_datasets

# %%
phase_train_datasets, phase_test_datasets = create_tf_dataset(client_ids, 'Y_train_Context', True, PHASE_CLASSES)

# %%
def make_client_data(datasets):
  client_data = {}
  # loop through the datasets
  for dataset in datasets:
      # get the session name
      session = dataset[0]

      # get the session data
      session_data = dataset[1]

      # add the session data to the client_data dictionary
      client_data[session] = session_data
  return client_data

# %%
# Make everything into a map for creating ClientData objects necessary for TF federated learning

phase_train_client_data = make_client_data(phase_train_datasets)
phase_test_client_data = make_client_data(phase_test_datasets)

# %%
client_ids = list(client_ids)

# %%
len(client_ids)

# %% [markdown]
# ## Setup Federated Data

# %%
def make_federated_data(client_data, client_ids):
  # Need a function to get the client data in order to make ClientData object
  def get_client_dataset(client_id):
    return client_data[client_id]

  # use tff to create ClientData object from our training data
  federated_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(client_ids, get_client_dataset)

  return federated_data

# %%
phase_train_federated_data = make_federated_data(phase_train_client_data, client_ids)
phase_test_federated_data = make_federated_data(phase_test_client_data, client_ids)

# %%
def preprocess(dataset, num_classes):
    def batch_format_fn(image, label):
        """Prepare a batch of data and return a (features, label) tuple."""
        batch_size = tf.shape(image)[0]  # Get the current batch size
        return (tf.reshape(image, [batch_size, 50, 50, 3]),
                tf.reshape(label, [batch_size, num_classes]))

    return dataset.batch(BATCH_SIZE).map(batch_format_fn)

def preprocess_federated_data(federated_data, num_classes):
  client_ids = sorted(federated_data.client_ids)[:NUM_CLIENTS]
  print(client_ids)
  federated_data = [preprocess(federated_data.create_tf_dataset_for_client(x), num_classes)
                          for x in client_ids]
  return federated_data

# %%
phase_train = preprocess_federated_data(phase_train_federated_data, PHASE_CLASSES)

# %%
phase_train[0]

# %% [markdown]
# ## Setup Model

# %% [markdown]
# ### Functions from Dandan's Code

# %%
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

        # if K.image_data_format() == 'channels_last':

        #     ROW_AXIS = 1; COL_AXIS = 2;  CHANNEL_AXIS = 3
        # else:
        #     CHANNEL_AXIS = 1; ROW_AXIS = 2; COL_AXIS = 3

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
        pool2_out = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1), name = "conv_final")(block)

        #flatten1 = keras.layers.GlobalAveragePooling2D(name = "GAP")(pool2)
        #out = keras.layers.Dense(num_outputs,activation='softmax')(pooled)

        flatten1 = Flatten( name = "Flatten")(pool2_out)

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


# %% [markdown]
# ### H1

# %%
# It makes the implementation simpler to remove any arguments from the model building function
def build_h1_model():
  return buildmodel((img_channels, img_rows, img_cols), PHASE_CLASSES,False)

# %%
# View the model architecture
model = build_h1_model()

# %%
# View the model architecture
model = build_h1_model()
# plot_model(model, expand_nested=True, dpi=60, show_shapes=True)

# %%
# model.summary()

# %% [markdown]
# ## Setup Federated Learning

# %%
def model_fn_h1():
  model = build_h1_model()
  return tff.learning.models.from_keras_model(
      model,
      input_spec=phase_train[0].element_spec,
      loss=tf.keras.losses.CategoricalCrossentropy(),
      metrics=[tf.keras.metrics.Accuracy()])

# %%
# @tff.tf_computation
def server_init_h1():
  model = model_fn_h1()
  return model.trainable_variables

# %%
@tff.federated_computation
def initialize_fn_h1():
  return tff.federated_value(server_init_h1(), tff.SERVER)

# %%
@tf.function
def client_update_h1(model, dataset, server_weights, client_optimizer):
  """Performs training (using the server model weights) on the client's dataset."""
  # Initialize the client model with the current server weights.
  client_weights = model.trainable_variables
  # Assign the server weights to the client model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        client_weights, server_weights)

  # Use the client_optimizer to update the local model.
  for batch in dataset:
    with tf.GradientTape() as tape:
      # Compute a forward pass on the batch of data
      outputs = model.forward_pass(batch)

    # Compute the corresponding gradient
    grads = tape.gradient(outputs.loss, client_weights)
    grads_and_vars = zip(grads, client_weights)

    # Apply the gradient using a client optimizer.
    client_optimizer.apply_gradients(grads_and_vars)

  return client_weights

# %%
@tf.function
def server_update_h1(model, mean_client_weights):
  """Updates the server model weights as the average of the client model weights."""
  model_weights = model.trainable_variables
  # Assign the mean client weights to the server model.
  tf.nest.map_structure(lambda x, y: x.assign(y),
                        model_weights, mean_client_weights)
  return model_weights

# %%
h1_fed = model_fn_h1()
tf_dataset_type_h1 = tff.SequenceType(h1_fed.input_spec)

# %%
str(tf_dataset_type_h1)

# %%
model_weights_type_h1 = h1_fed.trainable_variables
# Assuming model_weights_type is a list of trainable variables
model_weights_type_h1 = [v for v in model_weights_type_h1]

model_weights_type_h1 = tff.to_type([tf.TensorSpec.from_tensor(v.value()) for v in model_weights_type_h1])

# %%
@tff.tf_computation(tf_dataset_type_h1, model_weights_type_h1)
def client_update_fn_h1(tf_dataset, server_weights):
  model = model_fn_h1()
  client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
  return client_update_h1(model, tf_dataset, server_weights, client_optimizer)

# %%
@tff.tf_computation(model_weights_type_h1)
def server_update_fn_h1(mean_client_weights):
  model = model_fn_h1()
  return server_update_h1(model, mean_client_weights)

# %%
federated_server_type_h1 = tff.FederatedType(model_weights_type_h1, tff.SERVER)
federated_dataset_type_h1 = tff.FederatedType(tf_dataset_type_h1, tff.CLIENTS)

# %%
@tff.federated_computation(federated_server_type_h1, federated_dataset_type_h1)
def next_fn_h1(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = tff.federated_map(
      client_update_fn_h1, (federated_dataset, server_weights_at_client))

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn_h1, mean_client_weights)


  return server_weights

# %%
federated_algorithm_h1 = tff.templates.IterativeProcess(
    initialize_fn=initialize_fn_h1,
    next_fn=next_fn_h1
)

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# ### Creating test set

# %%
def create_test_set(federated_test_set, num_classes):
  # Create a list to store client datasets
  client_datasets = []

  # Iterate over client IDs and create datasets
  for client_id in sorted(federated_test_set.client_ids)[:NUM_CLIENTS]:
      client_dataset = federated_test_set.create_tf_dataset_for_client(client_id)
      client_datasets.append(client_dataset)

  # Combine the client datasets into a centralized dataset
  test_set = tf.data.experimental.sample_from_datasets(client_datasets)
  test_set = preprocess(test_set, num_classes)
  return test_set

# %%
phase_test_central = create_test_set(phase_test_federated_data, PHASE_CLASSES)
phase_train_central = create_test_set(phase_train_federated_data, PHASE_CLASSES)

# %% [markdown]
# ### Evaluation

# %% [markdown]
# We conduct an initial evaluation to ensure the previous steps have been successful.

# %%
# Model compile instructions taken from Supervisor's code
def evaluate_h1(server_state):
  model = build_h1_model()
  model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
  )
  model.set_weights(server_state)
  model.evaluate(phase_test_central)

# %%
server_state_h1 = federated_algorithm_h1.initialize()
# evaluate_h1(server_state_h1)

# %% [markdown]
# ## Build Eval Models (TFF glitch)

# %% [markdown]
# Due to the way tensorflow federated works, we will not be able to create new models once the training has begun. For this reason, we create the evaluaiton model before the training. This also allows us to track the performance of the model through training.

# %%
h1_eval = build_h1_model()
h1_eval.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
  )

# %% [markdown]
# ## Time Logging for training

# %%
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))

def calculate_time(round, start_time, times_taken, total_rounds):
    # Record the end time for the current round
    end_time = time.time()
    # Calculate the time taken for the current round
    time_taken = end_time - start_time
    # Append the time taken to the list of times
    times_taken.append(time_taken)

    # Calculate the average time taken for the previous rounds
    avg_time_taken = sum(times_taken) / len(times_taken)
    # Calculate the estimated remaining time for the remaining rounds
    remaining_rounds = total_rounds - (round + 1)
    estimated_remaining_time = remaining_rounds * avg_time_taken

    # Display the time taken for the current round and the estimated remaining time
    print('Time taken for round {:2d}: {}'.format(round, format_time(time_taken)))
    print('Estimated remaining time: {}'.format(format_time(estimated_remaining_time)))
    print('')

# %% [markdown]
# ## Model Training

# %% [markdown]
# ### Early stopping check

# %%
def early_stop_check(accuracy, best_accuracy, epochs_without_improvement):
  improvement = accuracy - best_accuracy
  if improvement > MIN_IMPROVEMENT:
      best_accuracy = accuracy
      epochs_without_improvement = 0
  else:
      epochs_without_improvement += 1
  # Stop training if no improvement for PATIENCE epochs
  if epochs_without_improvement >= PATIENCE:
      print("Early stopping: No improvement of at least {} for {} epochs.".format(MIN_IMPROVEMENT, PATIENCE))
      return True
  return False

# %% [markdown]
# ### Training Loop

# %%

import csv

# Define the early stopping parameters
PATIENCE = 10  # Number of epochs to wait for improvement
MIN_IMPROVEMENT = 0.001  # Minimum improvement threshold (adjust as needed)
best_accuracy = 0.0
epochs_without_improvement = 0

# Open the CSV file for writing
csv_file_path = 'logs' + '/' + LOGS_DIR + '/' + MODEL_NAME + '_train_test' + '.csv'
csv_columns = ['Round', 'Global_Train_Loss', 'Global_Train_Accuracy', 'Global_Test_Loss', 'Global_Test_Accuracy']  # Add more columns as needed

for client_id in client_ids:
    csv_columns.extend([f'{client_id}_Test_Loss', f'{client_id}_Test_Accuracy', f'{client_id}_Train_Loss', f'{client_id}_Train_Accuracy'])

# Write the CSV header
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    csv_writer.writeheader()

    times_taken = []
    for round in range(1, EPOCHS+1):
        start_time = time.time()

        # Set the server weights to be the result of performing federated averaging
        # on the client weights after one epoch of training.
        server_state_h1 = federated_algorithm_h1.next(server_state_h1, phase_train)

        # Conduct an evaluation of the training epoch
        h1_eval.set_weights(server_state_h1)
        global_train_loss, global_train_accuracy = h1_eval.evaluate(phase_train_central, verbose=0)
        global_test_loss, global_test_accuracy = h1_eval.evaluate(phase_test_central, verbose=0)
        print('Round {:2d}: \t global_test_loss={:.4f}, global_test_accuracy={:.4f} \n\t\tglobal_train_loss={:.4f}, global_train_accuracy={:.4f}'
              .format(round, global_test_loss, global_test_accuracy, global_train_loss, global_train_accuracy))

        # Early stopping check
        stop = early_stop_check(global_test_accuracy, best_accuracy, epochs_without_improvement)

        # Evaluate the model on a per-client basis
        client_metrics = {}
        for client_id in client_ids:
            client_dataset = phase_test_federated_data.create_tf_dataset_for_client(client_id)
            client_dataset = preprocess(client_dataset, PHASE_CLASSES)
            test_loss, test_accuracy = h1_eval.evaluate(client_dataset, verbose=0)

            client_dataset = phase_train_federated_data.create_tf_dataset_for_client(client_id)
            client_dataset = preprocess(client_dataset, PHASE_CLASSES)
            train_loss, train_accuracy = h1_eval.evaluate(client_dataset, verbose=0)

            # Save the metrics for each client
            client_metrics[f'{client_id}_Test_Loss'] = test_loss
            client_metrics[f'{client_id}_Test_Accuracy'] = test_accuracy
            client_metrics[f'{client_id}_Train_Loss'] = train_loss
            client_metrics[f'{client_id}_Train_Accuracy'] = train_accuracy

        # Save the metrics to CSV
        csv_row=({
            'Round': round,
            'Global_Train_Loss': global_train_loss,
            'Global_Train_Accuracy': global_train_accuracy,
            'Global_Test_Loss': global_test_loss,
            'Global_Test_Accuracy': global_test_accuracy,
            **client_metrics
        })

        csv_writer.writerow(csv_row)

        # Call the function to calculate time and display information
        calculate_time(round, start_time, times_taken, EPOCHS+1)

        # Stop if converged early
        if stop:
            print("Leaving training loop")
            round = EPOCHS



# %% [markdown]
# ## Post-Training Evaluation

# %%
h1_eval.set_weights(server_state_h1)
h1_eval.evaluate(phase_test_central)

# %%
h1_eval.save('Models/' + MODEL_NAME + '.h5')

# %%
# summary_writer_h1.close()

# %%
from google.colab import runtime
runtime.unassign()

# %%



