import logging
import time

import numpy as np
import pygad

from . import config, utils
from .evaluate import BudgetExhausted, Evaluator


def run_ga(splits, run_seed, budget=config.EVALUATION_BUDGET):
    tag = f"ga_seed{run_seed}"

    quiet = logging.getLogger(f"pygad_{run_seed}")
    quiet.addHandler(logging.NullHandler())
    quiet.propagate = False

    
    print(f"[{tag}] population {config.POPULATION_SIZE} | "
          f"up to {config.NUM_GENERATIONS} generations | budget {budget} trainings | "
          f"{config.PROXY_EPOCHS} proxy epochs")
    evaluator = Evaluator(splits, tag=tag, budget=budget)
    generation_log = []

    def fitness_func(ga_instance, solution, solution_idx):
        return evaluator(solution)


    def on_generation(ga_instance):
        evaluator.generation = int(ga_instance.generations_completed)
        _, best_fitness, _ = ga_instance.best_solution()
        generation_log.append({
            "generation": evaluator.generation,
            "best_fitness": float(best_fitness),
            "mean_fitness": float(np.mean(ga_instance.last_generation_fitness)),
            "std_fitness": float(np.std(ga_instance.last_generation_fitness)),
            "evaluations_spent": evaluator.n_evaluations,
        })
        print(f"[{tag}] generation {evaluator.generation}: best {best_fitness:.4f}, "
              f"spent {evaluator.n_evaluations}/{budget}")

    ga = pygad.GA(
        num_generations=config.NUM_GENERATIONS,
        num_parents_mating=config.NUM_PARENTS_MATING,
        fitness_func=fitness_func,
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
        on_generation=on_generation,
        random_seed=run_seed,
        suppress_warnings=True,
        logger=quiet,
    )

    start = time.time()
    stopped_early = False
    try:
        ga.run()
    except BudgetExhausted as exc:
        stopped_early = True
        print(f"[{tag}] {exc}, stopping mid-generation")

    summarise(evaluator, tag=tag, method="ga", run_seed=run_seed, budget=budget,
              elapsed=time.time() - start, generation_log=generation_log,
              stopped_early=stopped_early,
              extra={
                  "population_size": config.POPULATION_SIZE,
                  "generations_requested": config.NUM_GENERATIONS,
                  "generations_completed": int(ga.generations_completed),
                  "proxy_epochs": config.PROXY_EPOCHS,
                  "num_parents_mating": config.NUM_PARENTS_MATING,
                  "tournament_size": config.TOURNAMENT_SIZE,
                  "keep_elitism": config.KEEP_ELITISM,
              })



def summarise(evaluator, tag, method, run_seed, budget, elapsed, generation_log,
              stopped_early, extra):
    best = evaluator.best
    summary = {
        "tag": tag,
        "method": method,
        "run_seed": run_seed,
        "budget": budget,
        "evaluations_spent": evaluator.n_evaluations,
        "fitness_calls": len(evaluator.history),
        "cache_hits": len(evaluator.history) - evaluator.n_evaluations,
        "stopped_early": stopped_early,
        "best_fitness": best["val_macro_f1"],
        "best_genome_key": best["genome_key"],
        "best_params": {g: best[g] for g in ("learning_rate", "dropout", "batch_size",
                                             "filters_1", "filters_2", "filters_3")},
        "best_so_far_curve": evaluator.best_so_far_curve(),
        "generation_log": generation_log,
        "seconds": round(elapsed, 1),
        **extra,
    }
    utils.save_json(summary, config.LOG_DIR / f"search_{tag}.json")
    print(f"[{tag}] done in {elapsed / 60:.1f} min, {evaluator.n_evaluations} trainings, "
          f"best macro F1 {summary['best_fitness']:.4f} at {summary['best_genome_key']}")
