import csv
import time

import numpy as np
from scipy.stats import spearmanr

from . import config, train, utils

GENES = ["learning_rate", "dropout", "batch_size", "filters_1", "filters_2", "filters_3"]



def collect_evaluations():
    real = set()
    for path in sorted(config.LOG_DIR.glob("search_*.json")):
        summary = utils.load_json(path)
        if summary["budget"] == config.EVALUATION_BUDGET:
            real.add(summary["tag"])

    records = {}
    for path in sorted(config.LOG_DIR.glob("evaluations_*.csv")):
        tag = path.stem.replace("evaluations_", "")
        if tag not in real:
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["cached"].lower() == "true" or row["genome_key"] in records:
                    continue
                records[row["genome_key"]] = {
                    "genome_key": row["genome_key"],
                    "source": tag,
                    "proxy_macro_f1": float(row["val_macro_f1"]),
                    "learning_rate": float(row["learning_rate"]),
                    "dropout": float(row["dropout"]),
                    "batch_size": int(row["batch_size"]),
                    "filters_1": int(row["filters_1"]),
                    "filters_2": int(row["filters_2"]),
                    "filters_3": int(row["filters_3"]),
                    "params": int(row["params"]),  # for colouring the fidelity plot
                }
    return list(records.values())


# Draw n configurations spread evenly across the proxy score range
def stratified_sample(records, n=12, bands=3, seed=config.SEED):
    rng = np.random.default_rng(seed)
    ordered = sorted(records, key=lambda r: r["proxy_macro_f1"])
    chosen = []
    for band, chunk in enumerate(np.array_split(np.arange(len(ordered)), bands)):
        for i in rng.choice(chunk, size=n // bands, replace=False):
            chosen.append({**ordered[int(i)], "band": band})
    return chosen


# Fully train the sampled configurations and correlate against their proxy scores
def run(splits):
    sample = stratified_sample(collect_evaluations())
    results = []
    start = time.time()
    for i, entry in enumerate(sample, 1):
        params = {g: entry[g] for g in GENES}
        print(f"[{i}/{len(sample)}] {utils.format_params(params)} (proxy {entry['proxy_macro_f1']:.4f})")
        full = train.train_full(params, splits, tag=f"fidelity_{i:02d}", seed=config.SEED)
        results.append({
            **entry,
            "full_val_macro_f1": full["val_macro_f1"],
            "full_test_macro_f1": full["test_macro_f1"],
            "epochs_run": full["epochs_run"],
        })
        
        utils.save_json(results, config.LOG_DIR / "proxy_fidelity_partial.json")

    rho, p_value = spearmanr([r["proxy_macro_f1"] for r in results],
                             [r["full_val_macro_f1"] for r in results])
    utils.save_json({
        "n": len(results),
        "bands": 3,
        "proxy_epochs": config.PROXY_EPOCHS,
        "full_epochs": config.FINAL_EPOCHS,
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "results": results,
        "seconds": round(time.time() - start, 1),
    }, config.LOG_DIR / "proxy_fidelity.json")
    print(f"Spearman rho = {rho:.3f} (p = {p_value:.4f}, n = {len(results)})")
