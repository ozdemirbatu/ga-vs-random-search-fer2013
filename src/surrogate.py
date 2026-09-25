import logging
import time

import numpy as np
import pygad
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from . import config, proxy_fidelity, random_search, utils
from .evaluate import BudgetExhausted, Evaluator

FEATURES = ["log_lr", "dropout", "batch_size", "filters_1", "filters_2", "filters_3"]


def to_features(params):
    return [float(np.log10(params["learning_rate"])), float(params["dropout"]),
            float(params["batch_size"]), float(params["filters_1"]),
            float(params["filters_2"]), float(params["filters_3"])]


def build_surrogate(records, seed=config.SEED):
    X = np.array([to_features(r) for r in records])
    y = np.array([r["proxy_macro_f1"] for r in records])

    # 5-fold CV first
    rhos, errors = [], []
    for train_idx, test_idx in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        fold_model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        fold_model.fit(X[train_idx], y[train_idx])
        predicted = fold_model.predict(X[test_idx])
        rhos.append(spearmanr(predicted, y[test_idx]).statistic)
        errors.append(float(np.mean(np.abs(predicted - y[test_idx]))))

    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    model.n_jobs = 1

    validation = {
        "n_training_points": len(records),
        "cv_spearman_mean": float(np.mean(rhos)),
        "cv_spearman_std": float(np.std(rhos, ddof=1)),
        "cv_mean_absolute_error": float(np.mean(errors)),
        "feature_importance": dict(zip(FEATURES, model.feature_importances_.round(4).tolist())),
    }
    return model, validation


class SurrogateEvaluator(Evaluator):
    def __init__(self, model, tag):
        super().__init__(splits=None, tag=tag)
        self.model = model

    def log(self, row):
        pass

    def _train_and_score(self, genome, key):
        params = config.decode(genome)
        return {
            "run": self.tag, "tag": self.tag, "eval_index": len(self.history),
            "genome_key": key, **params,
            "val_macro_f1": float(self.model.predict([to_features(params)])[0]),
            "val_accuracy": float("nan"), "params": 0, "seconds": 0.0, "cached": False,
        }


def simulate_ga(model, seed):
    evaluator = SurrogateEvaluator(model, tag=f"sim_ga_{seed}")
    quiet = logging.getLogger(f"sim_{seed}")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False
    ga = pygad.GA(
        num_generations=config.NUM_GENERATIONS,
        num_parents_mating=config.NUM_PARENTS_MATING,
        fitness_func=lambda _g, sol, _i: evaluator(sol),
        sol_per_pop=config.POPULATION_SIZE,
        num_genes=len(config.GENE_NAMES),
        gene_space=config.gene_space_copy(),
        gene_type=config.gene_type_copy(),
        parent_selection_type=config.PARENT_SELECTION,
        K_tournament=config.TOURNAMENT_SIZE,
        crossover_type=config.CROSSOVER_TYPE,
        mutation_type=config.MUTATION_TYPE,
        mutation_probability=config.MUTATION_PROBABILITY,
        keep_elitism=config.KEEP_ELITISM,
        random_seed=seed,
        suppress_warnings=True,
        logger=quiet,
    )
    try:
        ga.run()
    except BudgetExhausted:
        pass
    return evaluator.best["val_macro_f1"]



def simulate_rs(model, seed):
    evaluator = SurrogateEvaluator(model, tag=f"sim_rs_{seed}")
    rng = np.random.default_rng(seed)
    while evaluator.n_evaluations < config.EVALUATION_BUDGET:
        evaluator(random_search.sample_genome(rng))
    return evaluator.best["val_macro_f1"]



def simulate(model, replications=500):
    start = time.time()
    ga_best, rs_best = [], []
    for seed in range(1000, 1000 + replications):
        ga_best.append(simulate_ga(model, seed))
        rs_best.append(simulate_rs(model, seed))
        if (seed - 999) % 25 == 0:
            print(f"  {seed - 999}/{replications} replications ({time.time() - start:.0f}s)")

    ga_best, rs_best = np.array(ga_best), np.array(rs_best)
    difference = ga_best - rs_best
    pooled = np.sqrt((ga_best.var(ddof=1) + rs_best.var(ddof=1)) / 2)
    statistic, p_value = mannwhitneyu(ga_best, rs_best, alternative="two-sided")
    return {
        "replications": replications,
        "budget": config.EVALUATION_BUDGET,
        "ga_mean": float(ga_best.mean()),
        "ga_std": float(ga_best.std(ddof=1)),
        "rs_mean": float(rs_best.mean()),
        "rs_std": float(rs_best.std(ddof=1)),
        "mean_difference": float(difference.mean()),
        "ga_win_rate": int((difference > 0).sum()) / replications,
        "ties": int((difference == 0).sum()),
        "cohens_d": float(difference.mean() / pooled),
        "mannwhitney_u": float(statistic),
        "mannwhitney_p": float(p_value),
        "ga_best_values": ga_best.round(5).tolist(),
        "rs_best_values": rs_best.round(5).tolist(),
        "seconds": round(time.time() - start, 1),
    }



def run(replications=500):
    records = proxy_fidelity.collect_evaluations()
    model, validation = build_surrogate(records)
    print(f"Surrogate on {len(records)} evaluations, CV Spearman "
          f"{validation['cv_spearman_mean']:.3f} ± {validation['cv_spearman_std']:.3f}")

    simulation = None
    if validation["cv_spearman_mean"] >= 0.5:
        simulation = simulate(model, replications)
        print(f"GA {simulation['ga_mean']:.4f} ± {simulation['ga_std']:.4f}, random search "
              f"{simulation['rs_mean']:.4f} ± {simulation['rs_std']:.4f}, GA ahead in "
              f"{simulation['ga_win_rate']:.1%} of replications")
    else:
        print("CV Spearman below 0.5, the simulation is not run")

    utils.save_json({"validation": validation, "simulation": simulation},
                    config.LOG_DIR / "surrogate.json")
