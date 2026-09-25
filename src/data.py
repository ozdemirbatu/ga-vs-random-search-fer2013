import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from . import config

ARRAY_CACHE = config.DATA_DIR / "fer2013_arrays.npz"
IMAGE_DIR = config.DATA_DIR / "fer2013"



def load_splits():
    if ARRAY_CACHE.exists():
        blob = np.load(ARRAY_CACHE)
        return {s: (blob[f"{s}_x"], blob[f"{s}_y"]) for s in ("train", "val", "test")}

    splits = decode_images()
    np.savez_compressed(ARRAY_CACHE,
                        train_x=splits["train"][0], train_y=splits["train"][1],
                        val_x=splits["val"][0], val_y=splits["val"][1],
                        test_x=splits["test"][0], test_y=splits["test"][1])
    return splits



def decode_images():

    label_of = {name: i for i, name in enumerate(config.EMOTIONS)}
    split_of = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
    found = {"train": [], "val": [], "test": []}

    for folder in sorted(IMAGE_DIR.rglob("*")):
        if not folder.is_dir() or folder.name not in label_of:
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            usage = path.name.split("_", 1)[0]
            with Image.open(path) as img:
                pixels = np.asarray(img.convert("L"), dtype=np.uint8)
            found[split_of[usage]].append((pixels, label_of[folder.name]))

    splits = {}
    for name, rows in found.items():
        images = np.stack([p for p, _ in rows])[..., None]
        labels = np.array([y for _, y in rows], dtype=np.int32)
        splits[name] = (images, labels)
    return splits


# Stratified fraction of each split for quick local test runs
def subsample(splits, fraction, seed=config.SEED):
    rng = np.random.default_rng(seed)
    reduced = {}
    for name, (images, labels) in splits.items():
        keep = []
        for cls in range(config.NUM_CLASSES):
            idx = np.where(labels == cls)[0]
            n = max(1, round(len(idx) * fraction))
            keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
        keep = np.sort(np.concatenate(keep))
        reduced[name] = (images[keep], labels[keep])
    return reduced


# Class counts per split
def print_class_distribution(splits):
    print("class     " + "".join(f"{name:>8}" for name in splits))
    for i, emotion in enumerate(config.EMOTIONS):
        counts = "".join(f"{int((labels == i).sum()):>8}" for _, labels in splits.values())
        print(f"{emotion:<10}{counts}")


# tf.data pipeline, pixels scaled to [0, 1]
def make_dataset(images, labels, batch_size, training, seed=config.SEED):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        ds = ds.shuffle(len(labels), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augment = augmenter()
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# Train plus clean val and test
def make_datasets(splits, batch_size, seed=config.SEED):
    return (
        make_dataset(*splits["train"], batch_size=batch_size, training=True, seed=seed),
        make_dataset(*splits["val"], batch_size=batch_size, training=False, seed=seed),
        make_dataset(*splits["test"], batch_size=batch_size, training=False, seed=seed),
    )


# Light augmentation 
def augmenter():
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.06),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )
