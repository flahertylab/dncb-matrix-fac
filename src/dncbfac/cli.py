import os
import click
import numpy as np

from bgnmf import BGNMF
from dncbtd import DNCBTD, initialize_DNCBTD_with_BGNMF

@click.command()

# Required Options
@click.option('--data_path', type=click.File('r'), required=True)
@click.option('--output_path', type=click.Path(), required=True)
@click.option('C','--C', type=int, required=True)
@click.option('K', '--K', type=int, required=True)

# Model Options
@click.option('--n_burnin', type=int, default=4)
@click.option('--n_epochs', type=int, default=3)
@click.option('--n_itns', type=int, default=2)

# Optional Options
@click.option('--seed', type=int, default=617, show_default=True)
@click.option('--pi_prior', type=str, default='gamma', show_default=True)
@click.option('--phi_prior', type=str, default='gamma', show_default=True)
@click.option('--theta_prior', type=str, default='gamma', show_default=True)
@click.option('--shp_pi', type=float, default=0.1, show_default=True)
@click.option('--shp_phi', type=float, default=0.1, show_default=True)
@click.option('--shp_theta', type=float, default=0.1, show_default=True)
@click.option('--shp_delta', type=float, default=0.1, show_default=True)
@click.option('--rte_delta', type=float, default=0.1, show_default=True)
@click.option('--bm', type=float, default=1, show_default=True)
@click.option('--bu', type=float, default=1, show_default=True)

# Running Options
@click.option('-v', '--verbose', count=True, help="Use multiple for increased verbosity.")
# @click.option('/debug;/no-debug')
@click.option('--n_threads', type=int, default=4, show_default=True)

def main(data_path, output_path, C, K,
         n_burnin, n_epochs, n_itns, seed, pi_prior, phi_prior,
         theta_prior, shp_pi, shp_phi, shp_theta, shp_delta, rte_delta, bm, bu,
         verbose, n_threads):
    
    # Load the data
    #import pdb
    #pdb.set_trace()

    data_dict = np.load(data_path.name)
    data_IJ = np.ascontiguousarray(data_dict['Beta_IJ'])
    I,J = data_IJ.shape

    # Instantiate the Model
    prbf_model = DNCBTD(I=I, J=J, C=C, K=K, bm=bm, bu=bu, pi_prior=pi_prior,   
                        phi_prior=phi_prior, theta_prior=theta_prior, 
                        shp_delta=shp_delta, rte_delta=rte_delta, 
                        shp_theta=shp_theta, shp_phi=shp_phi, shp_pi=shp_pi, 
                        debug=0, seed=seed, n_threads=n_threads)

    # Initialize the Model
    initialize_DNCBTD_with_BGNMF(prbf_model, data_IJ,  verbose=verbose, n_itns=5)

    # Create the output path
    samples_path = os.path.join(output_path, 'samples')
    os.makedirs(samples_path)

    # Fit the Model
    for epoch in range(n_epochs+2):
        if epoch > 0:
            prbf_model.fit(data_IJ = data_IJ, 
                           n_itns=n_itns if epoch > 1 else n_burnin,
                           verbose=verbose,
                           initialize=False,
                           schedule={},
                           fix_state={}
                           )

        state = dict(prbf_model.get_state())
        Theta_IC = state['Theta_IC']
        Phi_KJ = state['Phi_KJ']
        Pi_2CK = state['Pi_2CK']
        Pi_CK = Pi_2CK[0, :, :]       # clusters (C) x pathways (K) core matrix
        Y_IC = state['Y_IC']
        Y_KJ = state['Y_KJ']
        Y_2CK = state['Y_2CK']
        Y_2IJ = state['Y_2IJ']

    # Write the Model parameters
    state_name = f"state_{prbf_model.total_itns}.npz"
    np.savez_compressed(os.path.join(samples_path,state_name), Theta_IC = Theta_IC, Phi_KJ = Phi_KJ, Pi_CK = Pi_CK, Y_IC = Y_IC, Y_KJ = Y_KJ, Y_2CK = Y_2CK, Y_2IJ = Y_2IJ)
    
if __name__ == '__main__':
    main()