import sys
import subprocess
from datetime import datetime
from pathlib import Path
import os

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.2", config_path="config", config_name="config")
def run_all(cfg: DictConfig):
    """
    Runs h1.py, h2.py, and h3.py in sequence, placing their outputs
    under a single top-level directory that includes a timestamp.
    """

    # 1) Create a timestamped output directory under cfg.paths.output_dir
    timestamp_dir = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    output_dir = Path(cfg.paths.output_dir)
    base_out_dir = os.path.join(output_dir, timestamp_dir)
    os.makedirs(base_out_dir, exist_ok=True)

    # 2) Optionally save config if combined.save_cfg == true
    config_save_path = os.path.join(base_out_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 3) Run H1
    #    Pass combined.combined_output_dir and combined.save_cfg=false
    cmd_h1 = [
        "python",
        "-u",
        "h1.py",
        f"combined.combined_output_dir={timestamp_dir}",
        "combined.save_cfg=false"
    ]
    print(f"Running H1 with command:\n{' '.join(map(str, cmd_h1))}\n")
    subprocess.run(cmd_h1, check=True)

    # 4) Run H2
    cmd_h2 = [
        "python",
        "-u",
        "h2.py",
        f"combined.combined_output_dir={timestamp_dir}",
        "combined.save_cfg=false"
    ]
    print(f"Running H2 with command:\n{' '.join(map(str, cmd_h2))}\n")
    subprocess.run(cmd_h2, check=True)

    # 5) Run H3
    cmd_h3 = [
        "python",
        "-u",
        "h3.py",
        f"combined.combined_output_dir={timestamp_dir}",
        "combined.save_cfg=false"
    ]
    print(f"Running H3 with command:\n{' '.join(map(str, cmd_h3))}\n")
    subprocess.run(cmd_h3, check=True)

    print("All hierarchies (H1, H2, H3) completed successfully!")


if __name__ == "__main__":
    run_all()
