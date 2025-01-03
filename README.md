# Federated Learning for Care Robotics

This repository contains an implementation of **federated learning** for a **hierarchical approach** to behavioral cloning in care robotics. By training locally and exchanging only model weights, we preserve user privacy and achieve competitive performance compared to a non-federated setup.


---

## Table of Contents


1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Repository Structure](#3-repository-structure)
4. [Usage](#4-usage)
5. [Technical Details](#5-technical-details)
6. [Further Reading](#6-further-reading)


---

## 1. Overview

The project implements a **three-tier hierarchy** (`H1`, `H2`, `H3`) for robotic control:

* **H1** (Classification #1)
* **H2** (Classification #2)
* **H3** (Regression)

These hierarchies build upon each other, which **improves explainability** by decomposing the learning process into interpretable stages. Meanwhile, **federated learning** ensures data stays local, mitigating privacy concerns.


---

## 2. Installation

### Environment


1. **Create a Conda environment** with Python 3.10:

```
conda create -n fxai_env python=3.10 conda activate fxai_env
```

2. **Install jaxlib**:
```
pip install jaxlib==0.3.15 -f https://storage.googleapis.com/jax-releases/jax_releases.htmlz
```

3. **Install TensorFlow Federated**:
```
pip install tensorflow-federated==0.58.0
```

4. **Install remaining libraries** as necessary (e.g., TensorFlow, Keras, OmegaConf).  
- The file `environment.yml` has partial environment specs, but some dependencies may need manual installation.

---

## 3. Repository Structure

Below is a **simplified** layout of important files:

```
. 
├── Data
│ └── Videos_Database_20_Robot_WebCam_50_overall_database.npz 
├── environment.yml 
├── README.md 
└── src 
  ├── config 
  │ └── config.yaml 
  ├── h1.py 
  ├── h2.py 
  ├── h3.py 
  ├── run_all.py 
  └── utils 
    ├── data_utils.py 
    ├── federated_utils.py 
    ├── logging_utils.py 
    ├── model_h1.py 
    ├── model_h2.py 
    ├── model_h3.py 
    └── training_utils.py
```


- **`h1.py`, `h2.py`, `h3.py`**: Each handles the federated learning process for its respective hierarchical model.  
- **`run_all.py`**: Script that runs H1 → H2 → H3 in sequence, ensuring each stage starts from the model produced by the previous one.  
- **`utils/`**: Contains supporting utilities for data loading, federated setup, logging, and model definitions.  
- **`config/config.yaml`**: Hydra configuration specifying model parameters, data paths, etc.

---

## 4. Usage

1. **Edit** `config/config.yaml` to adjust hyperparameters, paths, or other settings.
2. **Run** the entire pipeline:
```
cd src python run_all.py
```

- A timestamped directory will be created under `cfg.paths.output_dir`, with subfolders for `H1`, `H2`, and `H3`.
- Logs, CSV metrics, and final `.h5` models are saved accordingly.
3. **Check logs** in each subfolder to see training progress and final metrics.

*To run only one hierarchy, you can execute `python h1.py`, `python h2.py`, or `python h3.py` individually, but note that `h2.py` and `h3.py` expect existing model weights from the previous stage.*

---

## 5. Technical Details

- **Federated Learning**: Implemented via **TensorFlow Federated** (v0.58.0).  
- **Hierarchical Models**:  
- **H1** learns a first-level classification.  
- **H2** refines or adds a second-level classification task.  
- **H3** focuses on regression using the final layer from H2.  
- Each subsequent hierarchy loads and adapts the model from the previous stage.  
- **Explainability**: By splitting the modeling task into multiple phases, the internal states (like “phase” or “state”) become more interpretable.

---

## 6. Further Reading

A broader discussion of **Federated Learning in Care Robotics**, including experimental results and a detailed presentation, is available in a separate repository:
[**Federated-Learning-in-Care-Robotics**](https://github.com/hpfield/Federated-Learning-in-Care-Robotics)

Feel free to explore that for in-depth information, additional data, and advanced experimentation details.
