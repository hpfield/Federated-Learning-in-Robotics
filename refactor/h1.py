# h1.py
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

# Local utility imports
from utils.logging_utils import setup_logger
from utils.data_utils import (
    create_tf_dataset, make_client_data, make_federated_data, 
    preprocess_federated_data, create_test_set
)
from utils.model_h1 import build_h1_model
from utils.federated_utils import build_iterative_process
from utils.training_utils import train_federated_model

import tensorflow as tf
print(f"Will show empty list if GPU unavailable: {tf.config.list_physical_devices('GPU')}\n\n")


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def main(cfg: DictConfig):

    # 1. Create output_run_dir based on cli.combined_output_dir
    base_output = Path(cfg.paths.output_dir)

    # If we have a combined_output_dir passed via CLI, use that
    if cfg.combined.combined_output_dir is not None:
        # e.g. "outputs/my_combined_run"
        output_run_dir = base_output / cfg.combined.combined_output_dir / "H1"
    else:
        # Otherwise, create a run-specific subfolder
        timestamp_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        # e.g. "outputs/H1_23-10-03_14-51-12"
        output_run_dir = base_output / f"H1_{timestamp_str}"
        
    # 2. Make sure it exists
    output_run_dir.mkdir(parents=True, exist_ok=True)

    # 3. Optionally save config if combined.save_cfg == true
    if cfg.combined.save_cfg:
        config_save_path = output_run_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

    # 4. Initialize logger to write into output_run_dir/logs
    logs_dir = output_run_dir / "logs"
    logger = setup_logger(str(logs_dir))

    # 5. Proceed with the rest of your training code
    # -----------------------------------------------------

    # Resolve nb_classes based on Cup_Type
    if cfg.data.cup_type == 'Medium':
        nb_classes = 9
    elif cfg.data.cup_type == 'Big':
        nb_classes = 10
    else:
        nb_classes = 7

    # Load data from NPZ
    database_used = np.load(cfg.data.npz_name)
    sessions = database_used['Session']

    # Build list of unique client IDs
    client_ids = np.unique(sessions).tolist()
    logger.info(f"Total unique clients: {len(client_ids)}")

    # Create train/test datasets (list of [client_id, tf.data.Dataset])
    phase_train_datasets, phase_test_datasets = create_tf_dataset(
        client_ids=client_ids,
        sessions_array=sessions,
        database=database_used,
        db_key='Y_train_Context',
        categorical=True,
        categories=cfg.client.h1.phase_classes
    )

    # Convert them into dictionary form
    phase_train_client_data = make_client_data(phase_train_datasets)
    phase_test_client_data = make_client_data(phase_test_datasets)

    # Build TFF ClientData
    phase_train_federated_data = make_federated_data(
        phase_train_client_data, 
        client_ids, 
        tff
    )
    phase_test_federated_data = make_federated_data(
        phase_test_client_data, 
        client_ids, 
        tff
    )

    # Preprocess the TFF data
    phase_train = preprocess_federated_data(
        phase_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h1.phase_classes,
        tff
    )

    # Build a centralized test dataset
    phase_test_central = create_test_set(
        phase_test_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h1.phase_classes,
        tff
    )

    # (Optional) Build a centralized train dataset
    phase_train_central = create_test_set(
        phase_train_federated_data,
        cfg.federated.num_clients,
        cfg.client.batch_size,
        cfg.client.h1.phase_classes,
        tff
    )

    # Build Keras model for final evaluation
    h1_eval = build_h1_model(
        img_channels=cfg.client.img_channels,
        img_rows=cfg.client.img_rows,
        img_cols=cfg.client.img_cols,
        phase_classes=cfg.client.h1.phase_classes
    )
    h1_eval.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(phase_train) == 0:
        raise ValueError("No training data for the specified clients.")

    example_dataset = phase_train[0]
    def build_h1_model_func():
        return build_h1_model(
            cfg.client.img_channels, 
            cfg.client.img_rows, 
            cfg.client.img_cols, 
            cfg.client.h1.phase_classes
        )

    # Build the TFF iterative process for H1
    federated_algorithm_h1 = build_iterative_process(build_h1_model_func, example_dataset)

    # Construct model name and CSV path inside output_run_dir
    model_name = f"H1_{cfg.federated.num_clients}_{cfg.federated.rounds}"
    csv_file_path = output_run_dir / f"{model_name}_train_test.csv"
    
    # Perform federated training
    logger.info("Starting Federated Learning")
    server_state = train_federated_model(
        rounds=cfg.federated.rounds,
        federated_algorithm=federated_algorithm_h1,
        phase_train=phase_train,
        phase_train_central=phase_train_central,
        phase_test_central=phase_test_central,
        model_build=build_h1_model,
        client_ids=client_ids[:cfg.federated.num_clients],
        client_epochs=tf.cast(cfg.client.epochs, tf.int64),
        hierarchy='h1',
        csv_file_path=str(csv_file_path),
        patience=cfg.federated.patience,
        min_improvement=cfg.federated.min_improvement,
        cfg=cfg
    )

    # Final evaluation
    h1_eval.set_weights(server_state)
    final_test_loss, final_test_accuracy = h1_eval.evaluate(phase_test_central, verbose=0)
    logger.info(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.4f}")

    # Save final model inside output_run_dir
    models_save_path = output_run_dir / f"{model_name}.h5"
    h1_eval.save(str(models_save_path))
    logger.info(f"Model saved at: {models_save_path}")

    # Overwrite cfg with the saved model path to cfg.client.h1.model_path
    cfg.client.h1.model_path = str(models_save_path)
    with open("config/config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
