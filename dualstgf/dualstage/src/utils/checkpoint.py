import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch


class EpochCheckpointManager:
    """
    Handles per-epoch checkpointing and metric logging.
    Creates a timestamped run directory under the provided root directory.
    """

    def __init__(self, root_dir: str, prefix: str = "dualstage", run_name: Optional[str] = None):
        self.root_path = Path(root_dir).expanduser().resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

        timestamp = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_path = self.root_path / f"{prefix}_{timestamp}"
        # Avoid accidental reuse when timestamp collisions happen.
        counter = 1
        while self.run_path.exists():
            self.run_path = self.root_path / f"{prefix}_{timestamp}_{counter:02d}"
            counter += 1
        self.run_path.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.run_path / "metrics.csv"
        self.prefix = prefix
        self.header = ["epoch", "train_loss", "val_loss", "val_anom", "time_seconds", "model_path"]

    def save_epoch(
        self,
        epoch: int,
        model: torch.nn.Module,
        train_loss: float,
        val_loss: float,
        val_anom: float,
        elapsed_time: float,
        extra_state: Optional[Dict[str, float]] = None,
    ) -> Path:
        """
        Saves the model parameters and appends the epoch metrics to the CSV log.
        Returns the path to the written checkpoint file.
        """
        date_tag = datetime.now().strftime("%Y%m%d")
        filename = f"{self.prefix}_{date_tag}_epoch_{epoch:03d}.pt"
        checkpoint_path = self.run_path / filename

        state_dict = model.state_dict()
        cpu_state = {
            key: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
            for key, tensor in state_dict.items()
        }
        torch.save(cpu_state, checkpoint_path)

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "val_anom": f"{val_anom:.6f}",
            "time_seconds": f"{elapsed_time:.1f}",
            "model_path": checkpoint_path.name,
        }

        if extra_state:
            for key, value in extra_state.items():
                row[key] = value

        self._append_row(row)

        return checkpoint_path

    def _append_row(self, row: Dict[str, str]) -> None:
        file_exists = self.metrics_path.exists()
        with self.metrics_path.open("a", newline="") as csvfile:
            fieldnames = self._fieldnames_with(row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _fieldnames_with(self, keys: Iterable[str]) -> Iterable[str]:
        extended = list(self.header)
        for key in keys:
            if key not in extended:
                extended.append(key)
        return extended

