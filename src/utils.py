#small helpers

import csv
import json
import random

import numpy as np
import tensorflow as tf

from . import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def gpu_report():
    gpus = tf.config.list_physical_devices("GPU")
    names = [tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus]
    return f"TensorFlow {tf.__version__}, {len(gpus)} GPU: {', '.join(names) or 'none'}"


def append_csv(path, fieldnames, row):
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_json(obj, path):
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def format_params(params):
    return (f"lr={params['learning_rate']:.2e} drop={params['dropout']:.2f} "
            f"bs={params['batch_size']} filters={params['filters_1']}/"
            f"{params['filters_2']}/{params['filters_3']}")
