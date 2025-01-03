import tensorflow_federated as tff
import tensorflow as tf
from tqdm import tqdm

def model_fn(build_model_func, train_example):
    """
    Creates a TFF learning model based on the provided Keras model.
    """
    model = build_model_func()
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=train_example.element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()]
    )

@tf.function
def client_update(model, dataset, server_weights, client_optimizer, epochs):
    """
    Performs local training (using the server model weights) on each client's dataset.
    Trains for the specified number of epochs.
    """
    client_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

    # Repeat the dataset for the specified number of epochs
    dataset = dataset.repeat(tf.cast(epochs, tf.int64))
    # train_pbar = tqdm(range(1, len(dataset)+1), desc=f"Client train")
    for batch in dataset:
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)
        client_optimizer.apply_gradients(grads_and_vars)

    return client_weights

@tf.function
def server_update(model, mean_client_weights):
    """
    Updates the server model with the average of the client model weights.
    """
    model_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_client_weights)
    return model_weights

def build_iterative_process(build_model_func, example_dataset):
    """
    Build a TFF iterative process, inlining the TFF computations.
    """
    def create_model():
        return model_fn(build_model_func, example_dataset)

    @tff.tf_computation
    def server_init():
        model = create_model()
        return model.trainable_variables

    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value(server_init(), tff.SERVER)

    tf_dataset_type = tff.SequenceType(
        model_fn(build_model_func, example_dataset).input_spec
    )
    model_weights_type = model_fn(build_model_func, example_dataset).trainable_variables
    model_weights_type = [v for v in model_weights_type]
    model_weights_type = tff.to_type([
        tf.TensorSpec.from_tensor(v.value()) for v in model_weights_type
    ])

    @tff.tf_computation(tf_dataset_type, model_weights_type, tf.int32)
    def client_update_fn(tf_dataset, server_weights, epochs):
        model = create_model()
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        return client_update(model, tf_dataset, server_weights, client_optimizer, epochs)

    @tff.tf_computation(model_weights_type)
    def server_update_fn(mean_client_weights):
        model = create_model()
        return server_update(model, mean_client_weights)

    @tff.federated_computation(
        tff.FederatedType(model_weights_type, tff.SERVER),
        tff.FederatedType(tf_dataset_type, tff.CLIENTS),
        tff.FederatedType(tf.int32, tff.CLIENTS)
    )
    def next_fn(server_weights, federated_dataset, federated_epochs):
        server_weights_at_client = tff.federated_broadcast(server_weights)
        client_weights = tff.federated_map(
            client_update_fn, (federated_dataset, server_weights_at_client, federated_epochs)
        )
        mean_client_weights = tff.federated_mean(client_weights)
        server_weights = tff.federated_map(server_update_fn, mean_client_weights)
        return server_weights

    return tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )
