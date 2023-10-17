## change lines 15, 34, 36, and 47 to reflect appropriate dataset name (ex. olivetti, array, predictionmarket) & percentage heldout & model version (eg. conditional vs not for dncb-td)
import os
import sys
import time
import numpy as np
from path import Path
from sbatch2 import sbatch
from argparse import ArgumentParser

# This is the path to your Python installation
PYTHON_INSTALL = '/usr/local/bin/python'

# This is the path to the experiment script you want to run
PYTHON_SCRIPT = Path('/work/src/run_heldout_experiment.py')

# # This is the path to the data -- not needed, since dataset is already specified in run_experiment_as.py
# OLIVETTI_FACES = Path('/home/gnagulpally/prbf_private/dat/faces/olivetti/data.npz')
# ARRAY_DATA = Path('/home/gnagulpally/prbf_private/dat/methylation/breast_ovary_colon_lung/data.npz')

def main():
    out_dir = Path('/home/gnagulpally/prbf_private/results/heldout_experiment')
    n_threads = 4
    for C in [6, 8, 10, 12, 14]:
        for K in [8, 10, 14, 20, 30]:
            for mask_seed in [413, 617, 781]:
                for model_seed in [413, 617, 781]:
                    for model in ['ggg']:
                        for lam in [275, 300, 400, 500, 600, 750, 1000]:
                        # for model in ['bdd', 'ggg', 'prbgnmf']:
                            if C > K:
                                continue

                            if model == 'prbgnmf':
                                params_path = '/home/gnagulpally/prbf_private/results/heldout_experiment/loci_5000/maskp_10/seed_{}/results/{}/K_{}/b_1.00/seed_{}/bisulfite_params.dat'.format(mask_seed, model, K, model_seed)
                            else:
                                params_path = '/home/gnagulpally/prbf_private/results/heldout_experiment/loci_5000/maskp_10/seed_{}/results/{}-cdncbtd/lam_{}/K_{}/C_{}/b_1.00/seed_{}/bisulfite_params.dat'.format(mask_seed, model, lam, K, C, model_seed)
                            cmd = [PYTHON_INSTALL,
                                    PYTHON_SCRIPT,
                                    '-p %s' % params_path]

                            cmd = ' '.join(cmd)

                            sbatch(cmd, 
                                    stdout=out_dir.joinpath('output.out'),  # where to save the output file
                                    stderr=out_dir.joinpath('errors.out'),
                                    n_threads=n_threads,                 
                                    job_name='bisulfite C%dK%d' % (C, K)           # job name; name it something descriptive
                            )
                                    
                            print(cmd)
                
        
if __name__ == '__main__':
    main()