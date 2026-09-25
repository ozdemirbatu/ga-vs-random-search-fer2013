
import random
import time

import keras
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import config, utils
from . import data as data_module
from . import model as model_module

class BudgetExhausted(Exception):
    pass


class Evaluator:
    FIELDS = ["run", "tag", "eval_index", "generation", "genome_key",
              "learning_rate", "dropout", "batch_size", "filters_1", "filters_2", "filters_3",
              "val_macro_f1", "val_accuracy", "params", "seconds", "cached"]

    def __init__(self, splits, tag, epochs=config.PROXY_EPOCHS, budget=config.EVALUATION_BUDGET):
        self.splits = splits
        self.tag = tag
        self.epochs = epochs
        self.budget = budget
        self.cache = {}
        self.history = []
        self.generation = 0
        self.log_path = config.LOG_DIR / f"evaluations_{tag}.csv"

    # Distinct models trained
    @property
    def n_evaluations(self):
        return len(self.cache)

    @property
    def best(self):
        return max(self.cache.values(), key=lambda r: r["val_macro_f1"])

    # Best fitness after each distinct training 
    def best_so_far_curve(self):
        curve, best = [], float("-inf")
        for record in self.history:
            if not record["cached"]:
                best = max(best, record["val_macro_f1"])
                curve.append(best)
        return curve

    def __call__(self, genome):
        key = config.genome_key(genome)
        if key in self.cache:
            record = dict(self.cache[key], cached=True, eval_index=len(self.history))
        else:
            if self.n_evaluations >= self.budget:
                raise BudgetExhausted(f"{self.tag}: spent all {self.budget} trainings")
            record = self._train_and_score(genome, key)
            self.cache[key] = record

        self.history.append(record)
        self.log({**record, "generation": self.generation})
        return record["val_macro_f1"]

    def log(self, row):
        utils.append_csv(self.log_path, self.FIELDS, row)

    def _train_and_score(self, genome, key):
        params = config.decode(genome)
        start = time.time()

        py_state, np_state = random.getstate(), np.random.get_state()
        try:
            utils.set_seed(config.SEED)
            train_ds = data_module.make_dataset(*self.splits["train"],
                                                batch_size=params["batch_size"], training=True)
            val_ds = data_module.make_dataset(*self.splits["val"],
                                              batch_size=params["batch_size"], training=False)
            net = model_module.build_model(params)
            net.fit(train_ds, epochs=self.epochs, verbose=0)
            macro_f1, accuracy = score(net, val_ds, self.splits["val"][1])
            n_params = int(net.count_params())
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

        elapsed = time.time() - start
        keras.backend.clear_session()

        print(f"[{self.tag} #{len(self.cache) + 1:>2}/{self.budget}] {utils.format_params(params)} "
              f"-> macroF1 {macro_f1:.4f} acc {accuracy:.4f} ({elapsed:.0f}s)")
        return {
            "run": self.tag,
            "tag": self.tag,
            "eval_index": len(self.history),
            "genome_key": key,
            **params,
            "val_macro_f1": float(macro_f1),
            "val_accuracy": float(accuracy),
            "params": n_params,
            "seconds": round(elapsed, 1),
            "cached": False,
        }


# Macro F1 and accuracy on a dataset
def score(net, dataset, true_labels):
    predictions = net.predict(dataset, verbose=0).argmax(axis=1)
    return (float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
            float(accuracy_score(true_labels, predictions)))


# Time one evaluation and extrapolate to the whole budget
def estimate_budget(splits):
    evaluator = Evaluator(splits, tag="probe")
    start = time.time()
    evaluator(config.BASELINE_GENOME)
    minutes = (time.time() - start) * config.EVALUATION_BUDGET / 60
    print(f"About {minutes:.0f} minutes ({minutes / 60:.1f} hours) per search")
