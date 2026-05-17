"""
logger.py
---------
Logger ligero que guarda métricas por episodio y las vuelca a CSV y a consola.
No requiere TensorBoard (aunque lo soporta opcionalmente).
"""

import csv
import os
import time
from collections import defaultdict


class MetricsLogger:
    def __init__(self, log_dir: str = "results", run_name: str = "iql"):
        self.log_dir  = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)

        self.csv_path = os.path.join(log_dir, f"{run_name}.csv")
        self._episode_data: dict = {}
        self._step_buffer: dict  = defaultdict(list)
        self._start_time  = time.time()
        self._episode     = 0
        self._fieldnames  = None

        # TensorBoard opcional
        self._tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(log_dir, "tensorboard", run_name)
            self._tb = SummaryWriter(tb_dir)
        except ImportError:
            pass

    # ------------------------------------------------------------------

    def log_step(self, info: dict, losses: dict, rewards: dict):
        """Llama en cada paso de simulación."""
        self._step_buffer["waiting"].append(info.get("total_waiting", 0))
        self._step_buffer["halted"].append(info.get("total_halted", 0))
        self._step_buffer["mean_waiting"].append(info.get("mean_waiting", 0))
        if losses:
            self._step_buffer["loss"].append(sum(losses.values()) / len(losses))
        if rewards:
            self._step_buffer["reward"].append(sum(rewards.values()))

    def log_episode(self, episode: int, extra: dict | None = None):
        """Llama al final de cada episodio."""
        self._episode = episode
        row = {"episode": episode,
               "elapsed_s": round(time.time() - self._start_time, 1)}

        for key, values in self._step_buffer.items():
            row[f"mean_{key}"] = round(sum(values) / max(len(values), 1), 4)
        if extra:
            row.update(extra)

        self._step_buffer.clear()

        # CSV
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writeheader()

        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fieldnames,
                           extrasaction="ignore").writerow(row)

        # TensorBoard
        if self._tb:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, episode)

        self._print(row)

    # ------------------------------------------------------------------

    def _print(self, row: dict):
        parts = [f"Ep {row['episode']:>4}"]
        for k in ("mean_mean_waiting", "mean_reward", "mean_loss"):
            if k in row:
                parts.append(f"{k.replace('mean_', '')}: {row[k]:.3f}")
        parts.append(f"t={row['elapsed_s']}s")
        print("  ".join(parts))

    def close(self):
        if self._tb:
            self._tb.close()
