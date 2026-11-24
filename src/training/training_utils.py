import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def env_str(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def configure_torch(seed=42):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() // 2)))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    return device


def cuda_status(device):
    if device.type != "cuda":
        return "CUDA: not available, using CPU"
    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3
    allocated_gb = torch.cuda.memory_allocated(device) / 1024**3
    return (
        f"CUDA: {props.name}, capability={props.major}.{props.minor}, "
        f"total={total_gb:.2f} GB, reserved={reserved_gb:.2f} GB, "
        f"allocated={allocated_gb:.2f} GB"
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_history_csv(path, rows, fieldnames=None):
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def should_log(step, every, total=None):
    every = max(1, every)
    return step % every == 0 or (total is not None and step == total)


class Timer:
    def __init__(self):
        self.start = time.time()

    @property
    def elapsed(self):
        return time.time() - self.start


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
