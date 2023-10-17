## example command: python test_script_dncbtd.py -o /home/gnagulpally/results_directory/

import sys
import numpy as np

from path import Path
from argparse import ArgumentParser

from bgnmf import BGNMF
from dncbtd import DNCBTD, initialize_DNCBTD_with_BGNMF

from IPython import embed

def main():
    p = ArgumentParser()
    p.add_argument('-o', '--out_dir', type=Path, required = True)
    p.add_argument('--seed', type=int, default=617)
    p.add_argument('--verbose', type=int, default=0)
    
    args = p.parse_args()
    
    data_dict = np.load('/data/methylation/sarcoma/data_top5k.npz')
    dataset_name = 'bisulfite'

    C = 4
    K = 6

    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    #data_IJ = np.ma.core.MaskedArray(Beta_IJ, mask=None) ## Not masking for this experiment
    I,J = data_IJ.shape

    pi_prior = 'gamma'     # each entry pi_ck^1 is drawn from a beta
    phi_prior = 'gamma'   # each column phi_j is drawn from a K-dimensional dirichlet
    theta_prior = 'gamma' # each row theta_i is drawn from a C-dimensional dirichlet

    shp_pi = 0.1      # hyperparam a_0^pi
    shp_phi = 0.1    # hyperparam a_0^phi
    shp_theta = 0.1   # hyperparam a_0^phi
    shp_delta = 0.1     # hyperparam a_0^delta
    rte_delta = 0.1  # hyperparam b_0^delta
    bm = 1
    bu = 1

    prbf_model = DNCBTD(I=I,
                        J=J,
                        C=C,
                        K=K,
                        bm=bm,
                        bu=bu,
                        pi_prior=pi_prior,
                        phi_prior=phi_prior,
                        theta_prior=theta_prior,
                        shp_delta=shp_delta,
                        rte_delta=rte_delta,
                        shp_theta=shp_theta,
                        shp_phi=shp_phi,
                        shp_pi=shp_pi,
                        debug=0,
                        seed=args.seed,
                        n_threads=4)

    initialize_DNCBTD_with_BGNMF(prbf_model, 
                                    data_IJ, 
                                    verbose=args.verbose,
                                    n_itns=5)
    
    args.out_dir.makedirs_p()
    samples_dir = args.out_dir.joinpath('samples')
    samples_dir.makedirs_p()

    n_burnin = 4
    n_epochs = 3
    n_itns = 2
    
    for epoch in range(n_epochs+2):
        if epoch > 0:
            prbf_model.fit(data_IJ = data_IJ,
                            n_itns=n_itns if epoch > 1 else n_burnin,
                            verbose=args.verbose,
                            initialize=False,
                            schedule={},
                            fix_state={})

        state = dict(prbf_model.get_state())
        Theta_IC = state['Theta_IC']
        Phi_KJ = state['Phi_KJ']
        Pi_2CK = state['Pi_2CK']
        Pi_CK = Pi_2CK[0, :, :]       # clusters (C) x pathways (K) core matrix
        Y_IC = state['Y_IC']
        Y_KJ = state['Y_KJ']
        Y_2CK = state['Y_2CK']
        Y_2IJ = state['Y_2IJ']


         
    np.savez_compressed(samples_dir.joinpath('state_%d_%s.npz' % (prbf_model.total_itns, dataset_name)), Theta_IC = Theta_IC, Phi_KJ = Phi_KJ, Pi_CK = Pi_CK, Y_IC = Y_IC, Y_KJ = Y_KJ, Y_2CK = Y_2CK, Y_2IJ = Y_2IJ)

if __name__ == '__main__':
    main()