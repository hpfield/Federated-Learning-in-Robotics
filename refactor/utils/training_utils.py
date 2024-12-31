# utilities/training_utils.py
import time
import csv
import tensorflow as tf
from tqdm import tqdm
import logging

logger = logging.getLogger("shared_logger")


def format_time(seconds):
    """
    Formats seconds into HH:MM:SS.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def calculate_time(round_idx, start_time, times_taken, total_rounds):
    """
    Appends time taken for the current round and calculates
    estimated remaining time based on average round time.
    """
    end_time = time.time()
    time_taken = end_time - start_time
    times_taken.append(time_taken)

    avg_time_taken = sum(times_taken) / len(times_taken)
    remaining_rounds = total_rounds - (round_idx + 1)
    estimated_remaining_time = remaining_rounds * avg_time_taken

    return time_taken, estimated_remaining_time


def train_federated_model(
    rounds,
    federated_algorithm,
    phase_train,
    phase_train_central,
    phase_test_central,
    model_build,
    client_ids,
    client_epochs,
    hierarchy,
    csv_file_path,
    patience,
    min_improvement,
    cfg
):
    """
    Main training loop for federated learning, with progress bar logging
    for training and evaluation, including estimated time remaining and
    global train accuracy.
    """
    server_state = federated_algorithm.initialize()
    best_accuracy = 0.0
    rounds_without_improvement = 0
    times_taken = []
    client_epochs_list = [client_epochs for _ in range(len(phase_train))]

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_columns = [
            'Round',
            'Global_Train_Loss',
            'Global_Train_Accuracy',
            'Global_Test_Loss',
            'Global_Test_Accuracy'
        ]
        for client_id in client_ids:
            csv_columns.extend([
                f'{client_id}_Test_Loss',
                f'{client_id}_Test_Accuracy',
                f'{client_id}_Train_Loss',
                f'{client_id}_Train_Accuracy'
            ])

        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        csv_writer.writeheader()

        with tf.device('/GPU:0'):
            pbar = tqdm(range(1, rounds + 1), desc="Training Rounds")
            for round_idx in pbar:
                start_time = time.time()

                # Perform one federated round
                server_state = federated_algorithm.next(server_state, phase_train, client_epochs_list)

                # Build a fresh model for evaluation each round
                if hierarchy=='h1':
                    new_eval_model = model_build(
                        cfg.client.img_channels,
                        cfg.client.img_rows,
                        cfg.client.img_cols,
                        cfg.client.h1.phase_classes
                    )
                    new_eval_model.compile(
                        loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                    )
                elif hierarchy=='h2':
                    new_eval_model = model_build()
                    new_eval_model.compile(
                        loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                    )
                elif hierarchy=='h3':
                    new_eval_model = model_build()
                    new_eval_model.compile(
                        loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['mse']
                    )
                new_eval_model.set_weights(server_state)

                # Evaluate on the global training set
                global_train_loss, global_train_accuracy = new_eval_model.evaluate(
                    phase_train_central, verbose=0)
                # Evaluate on the global test set
                global_test_loss, global_test_accuracy = new_eval_model.evaluate(
                    phase_test_central, verbose=0)

                # Early stopping check
                improvement = global_test_accuracy - best_accuracy
                if improvement > min_improvement:
                    best_accuracy = global_test_accuracy
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= patience:
                    logger.info(f"Early stopping at round {round_idx}.")
                    break

                # In-round evaluation progress
                eval_pbar = tqdm(client_ids, desc=f"Round {round_idx} Evaluation", leave=False)
                client_metrics = {}
                for client_id in eval_pbar:
                    client_dataset_test = phase_test_central
                    test_loss, test_accuracy = new_eval_model.evaluate(
                        client_dataset_test, verbose=0)
                    client_dataset_train = phase_train_central
                    train_loss, train_accuracy = new_eval_model.evaluate(
                        client_dataset_train, verbose=0)

                    client_metrics[f'{client_id}_Test_Loss'] = test_loss
                    client_metrics[f'{client_id}_Test_Accuracy'] = test_accuracy
                    client_metrics[f'{client_id}_Train_Loss'] = train_loss
                    client_metrics[f'{client_id}_Train_Accuracy'] = train_accuracy

                # Write metrics to CSV
                csv_row = {
                    'Round': round_idx,
                    'Global_Train_Loss': global_train_loss,
                    'Global_Train_Accuracy': global_train_accuracy,
                    'Global_Test_Loss': global_test_loss,
                    'Global_Test_Accuracy': global_test_accuracy,
                    **client_metrics
                }
                csv_writer.writerow(csv_row)

                # Timing info
                time_taken, estimated_remaining_time = calculate_time(
                    round_idx, start_time, times_taken, rounds)

                # Update progress bar with metrics
                pbar.set_postfix({
                    "Train_Acc": f"{global_train_accuracy:.4f}",
                    "Time_Remaining": format_time(estimated_remaining_time)
                })
    logger.info(f"Final results: \nGlobal_Train_Loss: {global_train_loss} \tGlobal_Train_Accuracy: {global_train_accuracy}\nGlobal_Test_Loss: {global_test_loss} \tGlobal_Test_Accuracy: {global_test_accuracy}")

    return server_state
