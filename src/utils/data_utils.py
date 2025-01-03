# utilities/data_utils.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging
logger = logging.getLogger("shared_logger")

def create_tf_dataset(
    client_ids, 
    sessions_array, 
    database, 
    db_key, 
    categorical, 
    categories, 
    test_size=0.2,
    random_state=100
):
    """
    Create a train/test split for each client and convert to tf.data.Dataset objects.
    """
    train_datasets = []
    test_datasets = []
    for session in client_ids:
        session_indices = np.where(sessions_array == session)[0]

        session_X = database['X_train'][session_indices]
        session_Y = database[db_key][session_indices]

        if categorical:
            session_Y = tf.keras.utils.to_categorical(session_Y, categories)

        X_train, X_test, Y_train, Y_test = train_test_split(
            session_X, session_Y, 
            test_size=test_size, 
            random_state=random_state
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        test_dataset  = tf.data.Dataset.from_tensor_slices((X_test,  Y_test))

        train_datasets.append([session, train_dataset])
        test_datasets.append([session, test_dataset])

    return train_datasets, test_datasets


def make_client_data(datasets):
    """
    Convert lists of (client_id, tf.data.Dataset) into a dictionary
    for easy consumption by TFF.
    """
    client_data = {}
    for dataset in datasets:
        session = dataset[0]
        session_data = dataset[1]
        client_data[session] = session_data
    return client_data


def make_federated_data(client_data, client_ids, tff):
    """
    Build a TFF ClientData object from the dictionary of client datasets.
    """
    def get_client_dataset(client_id):
        return client_data[client_id]

    federated_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids, get_client_dataset
    )
    return federated_data


def preprocess(dataset, batch_size, num_classes):
    """
    Prepare a batch of data (features, label) for input into the model.
    """
    def batch_format_fn(image, label):
        batch_size_dynamic = tf.shape(image)[0]
        return (
            tf.reshape(image, [batch_size_dynamic, 50, 50, 3]),
            tf.reshape(label, [batch_size_dynamic, num_classes])
        )

    return dataset.batch(batch_size).map(batch_format_fn)


def preprocess_federated_data(federated_data, num_clients, batch_size, num_classes, tff):
    """
    Preprocess each client's dataset into a list of preprocessed datasets.
    """
    client_ids = sorted(federated_data.client_ids)[:num_clients]
    logger.info(f"Using clients: {client_ids}")
    return [
        preprocess(federated_data.create_tf_dataset_for_client(x), batch_size, num_classes)
        for x in client_ids
    ]


def create_test_set(federated_test_set, num_clients, batch_size, num_classes, tff):
    """
    Build a centralized test set by combining all clients' test data.
    """
    client_datasets = []
    for client_id in sorted(federated_test_set.client_ids)[:num_clients]:
        client_dataset = federated_test_set.create_tf_dataset_for_client(client_id)
        client_datasets.append(client_dataset)

    combined_test_set = tf.data.experimental.sample_from_datasets(client_datasets)
    combined_test_set = preprocess(combined_test_set, batch_size, num_classes)
    return combined_test_set
