import numpy as np
import numpy.random as rn
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from bgnmf import BGNMF
from sklearn.decomposition import NMF
from IPython import embed


def initialize_PRBF_with_BGNMF(model, Beta_IJ, mask_IJ):
    params = model.get_params()
    C = params['C']
    K = params['K']

    print('Initializing with BGNMF...')

    if K >= C:
        bg_model = BGNMF(n_components=K, 
                         tol=1e-2, 
                         max_iter=500, 
                         verbose=False)

        bg_model.fit(Beta_IJ, mask_IJ.astype(bool))

        A_IK = bg_model.A_IK
        B_IK = bg_model.B_IK
        H_KJ = bg_model.H_KJ

        nmf_model = NMF(n_components=C)
        A_IC = nmf_model.fit_transform(A_IK)
        A_CK = nmf_model.components_

        B_IC = nmf_model.fit_transform(B_IK)
        B_CK = nmf_model.components_

        w_K = H_KJ.sum(axis=1)
        Phi_KJ = H_KJ / w_K[:, np.newaxis]

        Theta_IC = (A_IC + B_IC) / 2.
        w_I = Theta_IC.sum(axis=1)
        Theta_IC /= w_I[:, np.newaxis]

        Pi_2CK = np.stack([A_CK, B_CK])
        Pi_CK = Pi_2CK.sum(axis=0) + 1e-30
        Pi_2CK[0] /= Pi_CK
        Pi_2CK[1] = 1 - Pi_2CK[0]

        Mu_I = (A_IK.dot(H_KJ) + B_IK.dot(H_KJ)).sum(axis=1)
        zeta = (Mu_I / K).mean()

    else:
        bg_model = BGNMF(n_components=C, 
                         tol=1e-3, 
                         max_iter=20, 
                         verbose=False)

        bg_model.fit(Beta_IJ.T, mask_IJ.T.astype(bool))

        A_JC = bg_model.A_IK
        B_JC = bg_model.B_IK
        H_CI = bg_model.H_KJ

        nmf_model = NMF(n_components=K)
        A_JK = nmf_model.fit_transform(A_JC)
        A_KC = nmf_model.components_

        B_JK = nmf_model.fit_transform(B_JC)
        B_KC = nmf_model.components_

        Phi_JK = (A_JK + B_JK) / 2.
        w_K = Phi_JK.sum(axis=0)
        Phi_KJ = (Phi_JK / w_K).T

        Theta_IC = H_CI.T
        w_I = Theta_IC.sum(axis=1)
        Theta_IC /= w_I[:, np.newaxis]

        PI_2CK = np.stack([A_KC.T, B_KC.T])
        Pi_CK = Pi_2CK.sum(axis=0)
        Pi_2CK[0] /= Pi_CK
        Pi_2CK[1] = 1 - Pi_2CK[0]

        Mu_I = (A_JC.dot(H_CI) + B_JC.dot(H_CI)).sum(axis=0)
        zeta = (Mu_I / C).mean()
    
    fix_state = {}
    fix_state['Pi_2CK'] = Pi_2CK
    fix_state['Theta_IC'] = Theta_IC
    fix_state['Phi_KJ'] = Phi_KJ
    fix_state['zeta'] = zeta

    data_Beta_IJ = np.ma.core.MaskedArray(Beta_IJ, mask=mask_IJ)
    model.fit(Beta_IJ=data_Beta_IJ,
              n_itns=50,
              initialize=True,
              fix_state=fix_state,
              verbose=0)

    print('\n------------------\nInitialized.\n')
    model.reset_total_itns()

    # I, J = Beta_IJ.shape
    # Mu_2IJ = np.zeros((2,) + Beta_IJ.shape)
    # Mu_2IJ[0] = np.clip(model.A_IK.dot(model.H_KJ) - bm, a_min=0, a_max=None)
    # Mu_2IJ[1] = np.clip(model.B_IK.dot(model.H_KJ) - bu, a_min=0, a_max=None)
    # Y_2IJ = np.sqrt(np.floor(Mu_2IJ)).astype(np.int32)

    # A_IK = model.A_IK
    # B_IK = model.A_IK
    # H_KJ = model.H_KJ

    # return Y_2IJ

# def initialize_BNPPRBF(model, Beta_IJ, mask_IJ):
#     params = model.get_params()

#     print('Fitting BGNMF to initialize...')
#     Y_2IJ = initialize_with_BGNMF(Beta_IJ=Beta_IJ,
#                                         mask_IJ=mask_IJ,
#                                         C=params['C'],
#                                         K=params['K'])
    
#     sparsity = 100 * (1 - np.count_nonzero(Y_2IJ) / Y_2IJ.size)
#     print('Initialized Y_2IJ is %.2f percent sparse...' % sparsity)

#     fix_state = {}
#     fix_state['Y_2IJ'] = Y_2IJ
#     fix_state['Lambda_2IJ'] = np.sum(Y_2IJ, axis=0, dtype=float) + params['bm'], params['bu']
#     fix_state['delta_theta'] = 1
#     fix_state['delta_pi'] = 1

#     print('Initializing other params with Y_2IJ fixed...')
#     data_Beta_IJ = np.ma.core.MaskedArray(Beta_IJ, mask=mask_IJ)
#     model.fit(Beta_IJ=data_Beta_IJ,
#               n_itns=50,
#               initialize=True,
#               fix_state=fix_state,
#               verbose=10)

#     print('\n------------------\nInitialized.\n')
#     model.reset_total_itns()