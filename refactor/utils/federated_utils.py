# utilities/federated_utils.py
import tensorflow_federated as tff
import tensorflow as tf

def model_fn_h1(build_h1_model_func, phase_train_example):
    """
    Creates a TFF learning model based on the provided Keras model.
    """
    # Build a fresh Keras model
    model = build_h1_model_func()
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=phase_train_example.element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy()]
    )


@tff.tf_computation
def server_init_h1(model_fn):
    """
    Initializes the server weights for the H1 TFF model.
    """
    model = model_fn()
    return model.trainable_variables


@tff.federated_computation
def initialize_fn_h1(model_fn):
    """
    Creates the federated initialization function for H1.
    """
    return tff.federated_value(server_init_h1(model_fn), tff.SERVER)


@tf.function
def client_update_h1(model, dataset, server_weights, client_optimizer):
    """
    Performs local training (using the server model weights) on each client's dataset.
    """
    client_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

    for batch in dataset:
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)
        client_optimizer.apply_gradients(grads_and_vars)

    return client_weights


@tf.function
def server_update_h1(model, mean_client_weights):
    """
    Updates the server model with the average of the client model weights.
    """
    model_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_client_weights)
    return model_weights


# def build_iterative_process_h1(build_h1_model_func, phase_train_example):
#     """
#     Constructs a TFF iterative process for H1 using custom client/server update.
#     """
#     # Create a model function for TFF
#     def tff_model_fn():
#         return model_fn_h1(build_h1_model_func, phase_train_example)

#     tf_dataset_type = tff.SequenceType(
#         model_fn_h1(build_h1_model_func, phase_train_example).input_spec
#     )
#     model_weights_type = model_fn_h1(build_h1_model_func, phase_train_example).trainable_variables
#     model_weights_type = [v for v in model_weights_type]
#     model_weights_type = tff.to_type([
#         tf.TensorSpec.from_tensor(v.value()) for v in model_weights_type
#     ])

#     @tff.tf_computation(tf_dataset_type, model_weights_type)
#     def client_update_fn(tf_dataset, server_weights):
#         model = tff_model_fn()
#         client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#         return client_update_h1(model, tf_dataset, server_weights, client_optimizer)

#     @tff.tf_computation(model_weights_type)
#     def server_update_fn(mean_client_weights):
#         model = tff_model_fn()
#         return server_update_h1(model, mean_client_weights)

#     @tff.federated_computation(
#         tff.FederatedType(model_weights_type, tff.SERVER),
#         tff.FederatedType(tf_dataset_type, tff.CLIENTS)
#     )
#     def next_fn(server_weights, federated_dataset):
#         server_weights_at_client = tff.federated_broadcast(server_weights)
#         client_weights = tff.federated_map(
#             client_update_fn, (federated_dataset, server_weights_at_client)
#         )
#         mean_client_weights = tff.federated_mean(client_weights)
#         server_weights = tff.federated_map(server_update_fn, mean_client_weights)
#         return server_weights

#     # Build the IterativeProcess
#     iterative_process = tff.templates.IterativeProcess(
#         initialize_fn=initialize_fn_h1(tff_model_fn),
#         next_fn=next_fn
#     )
#     return iterative_process


def build_iterative_process_h1(build_h1_model_func, example_dataset):
    """
    Build a TFF iterative process, inlining the TFF computations.
    """

    def create_model():
        # This builds a TFF model from your Keras model
        return model_fn_h1(build_h1_model_func, example_dataset)

    @tff.tf_computation
    def server_init():
        # No 'model_fn' passed in as a parameter. We just call 'create_model()' directly.
        model = create_model()
        return model.trainable_variables

    @tff.federated_computation
    def initialize_fn():
        # Use the server_init() result as the server-side value
        return tff.federated_value(server_init(), tff.SERVER)

    # --------------------------------------------------
    # The rest of your TFF computations (client_update, server_update, next_fn, etc.)
    # --------------------------------------------------

    tf_dataset_type = tff.SequenceType(
        model_fn_h1(build_h1_model_func, example_dataset).input_spec
    )
    model_weights_type = model_fn_h1(build_h1_model_func, example_dataset).trainable_variables
    model_weights_type = [v for v in model_weights_type]
    model_weights_type = tff.to_type([
        tf.TensorSpec.from_tensor(v.value()) for v in model_weights_type
    ])

    @tff.tf_computation(tf_dataset_type, model_weights_type)
    def client_update_fn(tf_dataset, server_weights):
        # Rebuild a fresh Keras/TFF model
        model = create_model()
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        return client_update_h1(model, tf_dataset, server_weights, client_optimizer)

    @tff.tf_computation(model_weights_type)
    def server_update_fn(mean_client_weights):
        model = create_model()
        return server_update_h1(model, mean_client_weights)

    @tff.federated_computation(
        tff.FederatedType(model_weights_type, tff.SERVER),
        tff.FederatedType(tf_dataset_type, tff.CLIENTS)
    )
    def next_fn(server_weights, federated_dataset):
        server_weights_at_client = tff.federated_broadcast(server_weights)
        client_weights = tff.federated_map(
            client_update_fn, (federated_dataset, server_weights_at_client)
        )
        mean_client_weights = tff.federated_mean(client_weights)
        server_weights = tff.federated_map(server_update_fn, mean_client_weights)
        return server_weights

    return tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )
