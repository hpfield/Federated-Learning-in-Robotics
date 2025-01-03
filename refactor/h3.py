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
from tensorflow.keras.models import load_model, clone_model

# Local utility imports
from utils.logging_utils import setup_logger
from utils.data_utils import (
    create_tf_dataset, make_client_data, make_federated_data, 
    preprocess_federated_data, create_test_set
)
from utils.model_h3 import learn_model
from utils.federated_utils import build_iterative_process
from utils.training_utils import train_federated_model

print(f"Will show empty list if GPU unavailable: {tf.config.list_physical_devices('GPU')}\n\n")


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):

    # 1. Create output directory
    base_output = Path(cfg.paths.output_dir)
    if cfg.combined.combined_output_dir is not None:
        output_run_dir = base_output / cfg.combined.combined_output_dir / "H3"
    else:
        timestamp_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        output_run_dir = base_output / f"H3_{timestamp_str}"
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Optionally save config
    if cfg.combined.save_cfg:
        config_save_path = output_run_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # 3. Set up logger
    logs_dir = output_run_dir / "logs"
    logger = setup_logger(str(logs_dir))

    # 4. Load data
    database_used = np.load(cfg.data.npz_name)
    sessions = database_used['Session']
    client_ids = np.unique(sessions).tolist()
    logger.info(f"Total unique clients: {len(client_ids)}")

    # For regression, we assume there's a different key, e.g. "Y_train_Regression" or similar:
    h3_train_datasets, h3_test_datasets = create_tf_dataset(
        client_ids=client_ids,
        sessions_array=sessions,
        database=database_used,
        db_key="Y_train_Regress",    
        categorical=False,           
        categories=1                   
    )

    # Convert them into dictionary form
    h3_train_client_data = make_client_data(h3_train_datasets)
    h3_test_client_data = make_client_data(h3_test_datasets)

    # Build TFF ClientData
    h3_train_federated_data = make_federated_data(h3_train_client_data, client_ids, tff)
    h3_test_federated_data = make_federated_data(h3_test_client_data, client_ids, tff)

    # Preprocess
    # For regression, we might pass "phase_classes=1" or skip if your function allows
    h3_train = preprocess_federated_data(
        h3_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        1,   # For regression, 1 output
        tff
    )
    h3_test_central = create_test_set(
        h3_test_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        1,   # For regression
        tff
    )
    h3_train_central = create_test_set(
        h3_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        1,
        tff
    )

    if len(h3_train) == 0:
        raise ValueError("No training data for the specified clients.")

    example_dataset = h3_train[0]

    # -------------------------------------------------------------------
    # (A) Load pre-trained H2 model in an uncompiled state
    # -------------------------------------------------------------------
    logger.info(f"Loading pre-trained H2 model from: {cfg.client.h2.model_path}")
    h2_model_uncompiled = load_model(cfg.client.h2.model_path, compile=False)
    
    # -------------------------------------------------------------------
    # (B) Build an H3 model for regression
    # -------------------------------------------------------------------
    h3_pretrained = learn_model(
        oldmodel=h2_model_uncompiled,
        nb_classes=1,                  # single regression output
        Transfer_Type='Regression',
        summary=False
    )

    # -------------------------------------------------------------------
    # (C) Provide TFF an uncompiled clone
    # -------------------------------------------------------------------
    def build_h3_model_func():
        return tf.keras.models.clone_model(h3_pretrained)

    # Build the TFF iterative process for regression
    federated_algorithm_h3 = build_iterative_process(
        build_h3_model_func,
        example_dataset
    )

    # -------------------------------------------------------------------
    # (D) Perform Federated Training
    # -------------------------------------------------------------------
    model_name = f"H3_{cfg.federated.num_clients}_{cfg.federated.rounds}"
    csv_file_path = output_run_dir / f"{model_name}_train_test.csv"

    logger.info("Starting Federated Learning for H3 (Regression)")
    server_state = train_federated_model(
        rounds=cfg.federated.rounds,
        federated_algorithm=federated_algorithm_h3,
        phase_train=h3_train,
        phase_train_central=h3_train_central,
        phase_test_central=h3_test_central,
        model_build=build_h3_model_func,
        client_ids=client_ids[:cfg.federated.num_clients],
        client_epochs=tf.cast(cfg.client.epochs, tf.int64),
        hierarchy='h3',
        csv_file_path=str(csv_file_path),
        patience=cfg.federated.patience,
        min_improvement=cfg.federated.min_improvement,
        cfg=cfg
    )

    # -------------------------------------------------------------------
    # (E) Final local evaluation
    # -------------------------------------------------------------------
    # Build a fresh clone, set the final weights, compile for regression
    final_model = clone_model(h3_pretrained)
    final_model.set_weights(server_state)
    final_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    final_test_loss, final_test_mse = final_model.evaluate(h3_test_central, verbose=0)
    logger.info(f"Final Test Loss (MSE): {final_test_loss:.4f}, Final Test MSE: {final_test_mse:.4f}")

    # -------------------------------------------------------------------
    # (F) Save final model
    # -------------------------------------------------------------------
    models_save_path = output_run_dir / f"{model_name}.h5"
    final_model.save(str(models_save_path))
    logger.info(f"Model saved at: {models_save_path}")


if __name__ == "__main__":
    main()
