# DNCB Factorization

Source code for "Stable Dimensionality Reduction for Bounded Support Data" by Anjali N. Albert, Patrick Flaherty, and Aaron Schein.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make build' and 'make bash'
    ├── README.md          <- The top-level README for developers using this project.
    ├── requirements.txt   <- Dependencies          
    ├── notebooks           
    │
    ├── src                <- Source code for use in this project.
    │   ├── Makefile            
    │   │
    │   ├── scripts             <- Scripts to reproduce the real data experiments described in the paper
    │   │   └── make_stability_experiment_config.py
    │   │   └── submit_stability_experiment.j2
    │   │   └── make_heldout_experiment_config.py
    │   │   └── submit_heldout_experiment.j2
    │   │
    │   ├── dncbfac             <- Source code for the model python package
    │   │   └── cython_files        <- Model source code
    │   │   └── api.py          
    │   │   └── heldout_experiment.py
    │   │   └── stability_experiment.py
    │   │   └── bgnmf.py
    │   ├── tests               <- Tests 
    │


## Workflow

### Step 1: Configure model

The model environment is initialized via the command line using:
```
make build
```
An interactive terminal can be launched using:
```
make bash
```
To initialize an experiment, run:
```
python src/scripts/experiment_config_file.py
```
This will initialize a results directory containing a batch submission script.

### Step 2: Run model

Launch the experiment using:
```
sbatch results/experiment_name/submit_experiment_name.sh
```
Experiment results will be stored in the same directory.
