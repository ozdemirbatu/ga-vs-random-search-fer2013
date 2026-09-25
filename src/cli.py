# Command line entry point. Everything runs from a terminal because i was needed to run on colab

import argparse
import os
import sys

# Has to be set before tensorflow is imported
if "--quiet" in sys.argv:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from . import (analysis, config, data, evaluate, model, proxy_fidelity, random_search,
               search, surrogate, train, utils)


def load_splits(args):
    splits = data.load_splits()
    if args.subset < 1.0:
        splits = data.subsample(splits, args.subset)
    return splits


# Print versions, GPU, split sizes
def cmd_check(args):
    print("python", sys.version.split()[0])
    print("tensorflow", tf.__version__)
    print(utils.gpu_report())
    splits = load_splits(args)
    for name in ("train", "val", "test"):
        print(name, len(splits[name][1]))


# Class counts 
def cmd_data(args):
    data.print_class_distribution(load_splits(args))


# Model summary and parameter counts
def cmd_model(args):
    model.build_model(config.decode(config.BASELINE_GENOME)).summary()
    corners = {
        "smallest": [-4.0, 0.10, 32, 16, 32, 64],
        "baseline": config.BASELINE_GENOME,
        "largest": [-2.0, 0.60, 128, 64, 128, 256],
    }
    for name, genome in corners.items():
        print(name, config.genome_key(genome), model.count_params(config.decode(genome)))


# Time one evaluation and extrapolate the budget
def cmd_probe(args):
    evaluate.estimate_budget(load_splits(args))


# Train the mid range default configuration properly
def cmd_baseline(args):
    params = config.decode(config.BASELINE_GENOME)
    train.repeat_over_seeds(params, load_splits(args), tag="baseline", seeds=args.seeds)


# E1: one GA search per run seed
def cmd_ga(args):
    splits = load_splits(args)
    for run_seed in args.run_seeds:
        search.run_ga(splits, run_seed=run_seed, budget=args.budget)


# E2: one random search per run seed same budget as the GA
def cmd_rs(args):
    splits = load_splits(args)
    for run_seed in args.run_seeds:
        random_search.run_random_search(splits, run_seed=run_seed, budget=args.budget)


# E3: does the 3 epoch proxy rank configurations like full training does? (RQ2)
def cmd_fidelity(args):
    proxy_fidelity.run(load_splits(args))


# E4: full training of a search winner (--from ga_seed1 for one run, --from ga for the best run)
def cmd_final(args):
    path = config.LOG_DIR / f"search_{args.source}.json"
    if path.exists():
        summary = utils.load_json(path)
    else:
        runs = [utils.load_json(p)
                for p in sorted(config.LOG_DIR.glob(f"search_{args.source}_seed*.json"))]
        summary = max(runs, key=lambda r: r["best_fitness"])
    print(f"winner of {summary['tag']}: {utils.format_params(summary['best_params'])}")
    train.repeat_over_seeds(summary["best_params"], load_splits(args),
                            tag=f"final_{summary['tag']}", seeds=args.seeds)


# E5: fit a surrogate to the real evaluations and replay the comparison on it
def cmd_surrogate(args):
    surrogate.run()


# Build every figure and table the current results support
def cmd_analyse(args):
    analysis.run()


COMMANDS = {
    "check": cmd_check,
    "data": cmd_data,
    "model": cmd_model,
    "probe": cmd_probe,
    "baseline": cmd_baseline,
    "ga": cmd_ga,
    "rs": cmd_rs,
    "fidelity": cmd_fidelity,
    "final": cmd_final,
    "surrogate": cmd_surrogate,
    "analyse": cmd_analyse,
}


def main():
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=config.FINAL_SEEDS)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--budget", type=int, default=config.EVALUATION_BUDGET)
    parser.add_argument("--from", dest="source")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
