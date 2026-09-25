# Settings shared by all experiments

import os
from pathlib import Path

# THESIS_ROOT is my personal drive
PROJECT_ROOT = Path(os.environ.get("THESIS_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

for folder in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR, FIGURE_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ER2013
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTIONS)
INPUT_SHAPE = (48, 48, 1)

# Training seed
SEED = 42
FINAL_SEEDS = [42, 1337, 2024]

GENE_NAMES = ["log_learning_rate", "dropout", "batch_size", "filters_1", "filters_2", "filters_3"]
GENE_SPACE = [
    {"low": -4.0, "high": -2.0},  # learning rate 1e-4 to 1e-2
    {"low": 0.1, "high": 0.6},  # dropout
    [32, 64, 128],  # batch size
    [16, 32, 64],  # filters block 1
    [32, 64, 128],  # filters block 2
    [64, 128, 256],  # filters block 3
]
GENE_TYPE = [float, float, int, int, int, int]

# Mid range value for every gene 
BASELINE_GENOME = [-3.0, 0.35, 64, 32, 64, 128]

# Training
PROXY_EPOCHS = 3
FINAL_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10

# GA
POPULATION_SIZE = 8
NUM_GENERATIONS = 12  # its upper limit
NUM_PARENTS_MATING = 4
PARENT_SELECTION = "tournament"
TOURNAMENT_SIZE = 3
CROSSOVER_TYPE = "single_point"
MUTATION_TYPE = "random"
MUTATION_PROBABILITY = 0.2
KEEP_ELITISM = 1
EVALUATION_BUDGET = 48



def gene_space_copy():
    return [dict(spec) if isinstance(spec, dict) else list(spec) for spec in GENE_SPACE]


def gene_type_copy():
    return list(GENE_TYPE)


def decode(genome):
    return {
        "learning_rate": float(10.0 ** float(genome[0])),
        "dropout": float(genome[1]),
        "batch_size": int(genome[2]),
        "filters_1": int(genome[3]),
        "filters_2": int(genome[4]),
        "filters_3": int(genome[5]),
    }


# Cache key, lr and dropout are rounded
def genome_key(genome):
    p = decode(genome)
    return (f"lr{p['learning_rate']:.6g}_do{p['dropout']:.3f}_bs{p['batch_size']}"
            f"_f{p['filters_1']}-{p['filters_2']}-{p['filters_3']}")
