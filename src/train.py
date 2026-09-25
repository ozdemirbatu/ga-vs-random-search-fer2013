import time

import keras
import numpy as np
from sklearn.metrics import f1_score

from . import config, utils
from . import data as data_module
from . import evaluate as evaluate_module
from . import model as model_module


class MacroF1Checkpoint(keras.callbacks.Callback):
    def __init__(self, val_dataset, val_labels):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_labels = val_labels
        self.scores = []
        self.best_score = -1.0
        self.best_epoch = -1
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.val_dataset, verbose=0).argmax(axis=1)
        score = float(f1_score(self.val_labels, predictions, average="macro", zero_division=0))
        self.scores.append(score)
        logs["val_macro_f1"] = score
        improved = score > self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        print(f"  val_macro_f1: {score:.4f}" + ("  (best)" if improved else ""))


def train_full(params, splits, tag, seed=config.SEED):
    utils.set_seed(seed)
    start = time.time()

    train_ds, val_ds, test_ds = data_module.make_datasets(splits, batch_size=params["batch_size"],
                                                          seed=seed)
    net = model_module.build_model(params)
    monitor = MacroF1Checkpoint(val_ds, splits["val"][1])
    callbacks = [
        monitor,
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                                      restore_best_weights=False, verbose=2),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                                          min_lr=1e-6, verbose=2),
    ]
    history = net.fit(train_ds, epochs=config.FINAL_EPOCHS, validation_data=val_ds,
                      callbacks=callbacks, verbose=2)
    net.set_weights(monitor.best_weights)

    test_f1, test_accuracy = evaluate_module.score(net, test_ds, splits["test"][1])
    predictions = net.predict(test_ds, verbose=0).argmax(axis=1)
    per_class = f1_score(splits["test"][1], predictions, average=None,
                         labels=range(config.NUM_CLASSES), zero_division=0)
    elapsed = time.time() - start

    model_path = config.CHECKPOINT_DIR / f"{tag}_seed{seed}.keras"
    net.save(model_path)

    result = {
        "tag": tag,
        "seed": seed,
        "params": params,
        "epochs_run": len(history.history["loss"]),
        "best_epoch": monitor.best_epoch,
        "val_macro_f1": monitor.best_score,
        "test_macro_f1": test_f1,
        "test_accuracy": test_accuracy,
        "per_class_f1": {name: float(f) for name, f in zip(config.EMOTIONS, per_class)},
        "n_parameters": int(net.count_params()),
        "seconds": round(elapsed, 1),
        "model_path": str(model_path),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "val_macro_f1_history": monitor.scores,
    }
    utils.save_json(result, config.LOG_DIR / f"final_{tag}_seed{seed}.json")
    print(f"[{tag} seed {seed}] test macro F1 {test_f1:.4f}, accuracy {test_accuracy:.4f}, "
          f"{result['epochs_run']} epochs, {elapsed / 60:.1f} min")
    keras.backend.clear_session()
    return result



def repeat_over_seeds(params, splits, tag, seeds):
    runs = [train_full(params, splits, tag=tag, seed=s) for s in seeds]
    f1s = np.array([r["test_macro_f1"] for r in runs])
    accs = np.array([r["test_accuracy"] for r in runs])
    summary = {
        "tag": tag,
        "params": params,
        "seeds": list(seeds),
        "test_macro_f1_mean": float(f1s.mean()),
        "test_macro_f1_std": float(f1s.std(ddof=1)),
        "test_accuracy_mean": float(accs.mean()),
        "test_accuracy_std": float(accs.std(ddof=1)),
        "runs": runs,
    }
    utils.save_json(summary, config.LOG_DIR / f"summary_{tag}.json")
    print(f"[{tag}] over {len(seeds)} seeds: macro F1 {summary['test_macro_f1_mean']:.4f} "
          f"± {summary['test_macro_f1_std']:.4f}, accuracy {summary['test_accuracy_mean']:.4f} "
          f"± {summary['test_accuracy_std']:.4f}")
