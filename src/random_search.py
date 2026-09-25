
import time

import numpy as np

from . import config
from .evaluate import Evaluator
from .search import summarise


def sample_genome(rng):
    genome = []
    for spec, gene_type in zip(config.GENE_SPACE, config.GENE_TYPE):
        if isinstance(spec, dict):
            value = rng.uniform(spec["low"], spec["high"])
        else:
            value = spec[int(rng.integers(len(spec)))]
        genome.append(gene_type(value))
    return genome


def run_random_search(splits, run_seed, budget=config.EVALUATION_BUDGET):
    tag = f"rs_seed{run_seed}"
    print(f"[{tag}] budget {budget} trainings | {config.PROXY_EPOCHS} proxy epochs")
    evaluator = Evaluator(splits, tag=tag, budget=budget)

    
    rng = np.random.default_rng(run_seed)
    start = time.time()
    draws = 0
    while evaluator.n_evaluations < budget:
        evaluator(sample_genome(rng))
        draws += 1
        evaluator.generation = draws

    summarise(evaluator, tag=tag, method="random", run_seed=run_seed, budget=budget,
              elapsed=time.time() - start, generation_log=[], stopped_early=False,
              extra={"draws": draws, "proxy_epochs": config.PROXY_EPOCHS})
