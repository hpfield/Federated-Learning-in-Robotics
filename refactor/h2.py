import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import keras

# Local utility imports
from utils.logging_utils import setup_logger
from utils.data_utils import (
    create_tf_dataset, make_client_data, make_federated_data, 
    preprocess_federated_data, create_test_set
)

from utils.model_h2 import learn_model
from utils.federated_utils import build_iterative_process
from utils.training_utils import train_federated_model

from tensorflow.keras.models import load_model, clone_model

print(f"Will show empty list if GPU unavailable: {tf.config.list_physical_devices('GPU')}\n\n")


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):

    # 1. Create output_run_dir based on cfg.combined.combined_output_dir
    base_output = Path(cfg.paths.output_dir)

    if cfg.combined.combined_output_dir is not None:
        output_run_dir = base_output / cfg.combined.combined_output_dir / "H2"
    else:
        timestamp_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        output_run_dir = base_output / f"H2_{timestamp_str}"

    output_run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Optionally save config
    if cfg.combined.save_cfg:
        config_save_path = output_run_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # 3. Set up logger
    logs_dir = output_run_dir / "logs"
    logger = setup_logger(str(logs_dir))

    # 4. Possibly resolve nb_classes if needed for debugging
    if cfg.data.cup_type == 'Medium':
        nb_classes = 9
    elif cfg.data.cup_type == 'Big':
        nb_classes = 10
    else:
        nb_classes = 7

    # Load data
    database_used = np.load(cfg.data.npz_name)
    sessions = database_used['Session']
    client_ids = np.unique(sessions).tolist()
    logger.info(f"Total unique clients: {len(client_ids)}")

    # Create TF datasets
    state_train_datasets, state_test_datasets = create_tf_dataset(
        client_ids=client_ids,
        sessions_array=sessions,
        database=database_used,
        db_key='Y_train_State',
        categorical=True,
        categories=cfg.client.h2.state_classes
    )

    state_train_client_data = make_client_data(state_train_datasets)
    state_test_client_data = make_client_data(state_test_datasets)

    # TFF client data
    state_train_federated_data = make_federated_data(state_train_client_data, client_ids, tff)
    state_test_federated_data = make_federated_data(state_test_client_data, client_ids, tff)

    # Preprocess
    state_train = preprocess_federated_data(
        state_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h2.state_classes,
        tff
    )
    state_test_central = create_test_set(
        state_test_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h2.state_classes,
        tff
    )
    state_train_central = create_test_set(
        state_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h2.state_classes,
        tff
    )

    if len(state_train) == 0:
        raise ValueError("No training data for the specified clients.")

    example_dataset = state_train[0]

    # -----------------------------------------------------------
    #  (A) Load pre-trained H1 in an UNCOMPILED state
    # -----------------------------------------------------------
    logger.info(f"Loading pre-trained H1 model from: {cfg.client.h1.model_path}")
    h1_model_uncompiled = load_model(cfg.client.h1.model_path, compile=False)

    # -----------------------------------------------------------
    #  (B) Build an H2 model using your 'learn_model', but skip compile
    # -----------------------------------------------------------

    h2_pretrained = learn_model(
        oldmodel=h1_model_uncompiled,
        nb_classes=cfg.client.h2.state_classes,
        Transfer_Type='Classification',
        summary=False
    )

    # -----------------------------------------------------------
    #  (C) Provide TFF an uncompiled clone
    # -----------------------------------------------------------

    def build_h2_model_func():
        return keras.models.clone_model(h2_pretrained)

    federated_algorithm_h2 = build_iterative_process(build_h2_model_func, example_dataset)

    # -----------------------------------------------------------
    #  (D) Train with TFF
    # -----------------------------------------------------------
    model_name = f"H2_{cfg.federated.num_clients}_{cfg.federated.rounds}"
    csv_file_path = output_run_dir / f"{model_name}_train_test.csv"

    logger.info("Starting Federated Learning for H2")
    server_state = train_federated_model(
        rounds=cfg.federated.rounds,
        federated_algorithm=federated_algorithm_h2,
        phase_train=state_train,
        phase_train_central=state_train_central,
        phase_test_central=state_test_central,
        model_build=build_h2_model_func,
        client_ids=client_ids[:cfg.federated.num_clients],
        client_epochs=tf.cast(cfg.client.epochs, tf.int64),
        hierarchy='h2',
        csv_file_path=str(csv_file_path),
        patience=cfg.federated.patience,
        min_improvement=cfg.federated.min_improvement,
        cfg=cfg
    )

    # -----------------------------------------------------------
    #  (E) Final local evaluation
    # -----------------------------------------------------------
    final_model = clone_model(h2_pretrained)
    final_model.set_weights(server_state)
    # Now we can compile locally
    final_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    final_test_loss, final_test_accuracy = final_model.evaluate(state_test_central, verbose=0)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.4f}")

    # -----------------------------------------------------------
    #  (F) Save final model
    # -----------------------------------------------------------
    models_save_path = output_run_dir / f"{model_name}.h5"
    final_model.save(str(models_save_path))
    logger.info(f"Model saved at: {models_save_path}")

    # Overwrite cfg with the saved model path to cfg.client.h1.model_path
    cfg.client.h2.model_path = str(models_save_path)
    with open("config/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
