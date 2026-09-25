# GA vs random search for CNN hyperparameters on FER2013

Code for my MSc dissertation at Gisma University of Applied Sciences (M598).
The experiments were run on Google Colab Pro with a T4 GPU.

## Setup

```
pip install tensorflow scikit-learn scipy matplotlib pygad pillow
```

Put the FER2013 image folders (`train/` and `test/`) in `data/fer2013/`. Results are written to `results/`.

## Commands

```
python -m src.cli check
python -m src.cli baseline --seeds 42 1337 2024
python -m src.cli ga --run-seeds 1 2 3 4 5
python -m src.cli rs --run-seeds 1 2 3 4 5
python -m src.cli fidelity
python -m src.cli final --from ga
python -m src.cli final --from rs
python -m src.cli surrogate
python -m src.cli analyse
```
