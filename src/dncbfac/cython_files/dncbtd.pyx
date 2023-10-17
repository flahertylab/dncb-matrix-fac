# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
import scipy.stats as st
import scipy.special as sp
cimport numpy as np
from libc.math cimport sqrt, exp, log, log1p
from collections import defaultdict

from cython.parallel import parallel, prange
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads

from mcmc_model_parallel cimport MCMCModel
from sample cimport _sample_gamma, _sample_beta, _sample_dirichlet
from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _bessel_mode

from bgnmf import BGNMF
from sklearn.decomposition import NMF

from likelihood import dncb_pdf, dncb_pdf_mc, cdncb_pdf

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass


cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_rng_uniform(gsl_rng * r)
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)
    double gsl_ran_beta_pdf(double x, double a, double b)
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef extern from "gsl/gsl_sf_hyperg.h" nogil:
    double gsl_sf_hyperg_1F1(double a, double b, double x)

cdef extern from "gsl/gsl_sf_gamma.h" nogil:
    double gsl_sf_gamma(double x)
    double gsl_sf_lngamma(double x)


cdef class DNCBTD(MCMCModel):
    """Doubly Non-Central Beta Tucker Decomposition
    
    Parameters
    ----------
    ...

    pi_prior : str, optional, default is `beta'
               The prior over elements of the core matrix Pi.
               
               Options are ['beta', 'gamma'].
               if 'beta' (default):
                    pi_ck ~ Beta(shp_pi, shp_pi)
               else if 'gamma':
                    pi_ck ~ Gamma(shp_pi, 1/rte_pi)
               
               Setting pi_prior='gamma' is incompatible with some priors for theta/phi.
               Specifically, if theta_prior='dir_C' or phi_prior='dir_K' then
               setting pi_prior='gamma' will throw an error.

    theta_prior : str, optional, default is 'row-dir'
                  The prior over elements of the samples embedding matrix Theta.

                  Options are ['gamma', 'dir_C', 'dir_I'].
                  if 'gamma':
                        theta_ic ~ Gamma(shp_theta, 1/rte_theta)
                  else if 'dir_C' (default):
                        Each row i is drawn from a C-dimensional Dirichlet:
                        (theta_i1, ... ,theta_iC) ~ Dir(shp_theta, ..., shp_theta)
                        Note: this is imcompatible with pi_prior='gamma'.
                  else if 'dir-I':
                        Each column c is drawn from an I-dimensional Dirichlet:
                        (theta_1c, ... ,theta_Ic) ~ Dir(shp_theta, ..., shp_theta)
    
    phi_prior : str, optional, default is 'col-dir'
                  The prior over elements of the loci embedding matrix Phi.

                  Options are ['gamma', 'dir_K', 'dir-J'].
                  if 'gamma':
                        phi_kj ~ Gamma(shp_phi, 1/rte_phi)
                  else if 'dir_K' (default):
                        Each column j is drawn from a K-dimensional Dirichlet:
                        (phi_1j, ... ,phi_Kj) ~ Dir(shp_phi, ..., shp_phi)
                        Note: this is imcompatible with pi_prior='gamma'.
                  else if 'dir-J':
                        Each row k is drawn from a J-dimensional Dirichlet:
                        (phi_k1, ... ,phi_kJ) ~ Dir(shp_phi, ..., shp_phi)

    ...

    """

    cdef:
        int I, J, C, K, Y_, debug, any_missing, conditional
        double bm, bu, shp_delta, rte_delta, shp_theta, rte_theta, shp_phi, rte_phi, shp_pi, rte_pi, delta, shp_c, rte_c, lam
        double[::1] _zeta_C, _zeta_K, c_I, c_J
        double[:,::1] Theta_IC, Phi_KJ, Beta_IJ, _P_TC, Lambda_IJ
        double[:,::1] _shp_JK, _shp_KJ, _Phi_JK, _shp_IC, _shp_CI, _Theta_CI
        double[:,:,::1] Pi_2CK, Mu_2IJ, _P_TCK
        long[:,::1] Y_IC, Y_KJ
        long[:,:,::1] Y_2IJ, _Y_TKJ, Y_2CK
        long[:,:,:,::1] _Y_T2CK
        unsigned int[:,::1] _N_TC, is_obs_IJ
        unsigned int[:,:,::1] _N_TCK
        str pi_prior, theta_prior, phi_prior
    
    def __init__(self, int I, int J, int C, int K, double bm=1., double bu=1.,
                 str pi_prior='beta', str theta_prior='dir_C', str phi_prior='dir_J',
                 int conditional=0, double shp_theta=0.1, double rte_theta=0.1,
                 double shp_phi=0.1, double rte_phi=0.1, 
                 double shp_pi=0.1, double rte_pi=0.1,
                 double shp_delta=0.1, double rte_delta=0.1,
                 double shp_c=0.1, double rte_c=0.1, double lam=1.,
                 int debug=0, object seed=None, object n_threads=None):

        assert pi_prior in ['beta', 'gamma']
        assert phi_prior in ['dir_K', 'dir_J', 'gamma']
        assert theta_prior in ['dir_C', 'dir_I', 'gamma']
        if pi_prior == 'gamma':
            assert (phi_prior != 'dir_K') and (theta_prior != 'dir_C')

        if n_threads is None:
            n_threads = omp_get_max_threads()
            print('Max threads: %d' % n_threads)
            
        super(DNCBTD, self).__init__(seed=seed, n_threads=n_threads)

        # Params
        self.I = self.param_list['I'] = I
        self.J = self.param_list['J'] = J
        self.C = self.param_list['C'] = C
        self.K = self.param_list['K'] = K
        self.bm = self.param_list['bm'] = bm
        self.bu = self.param_list['bu'] = bu
        self.conditional = self.param_list['conditional'] = conditional
        self.pi_prior = self.param_list['pi_prior'] = pi_prior
        self.phi_prior = self.param_list['phi_prior'] = phi_prior
        self.theta_prior = self.param_list['theta_prior'] = theta_prior
        self.shp_theta = self.param_list['shp_theta'] = shp_theta
        self.rte_theta = self.param_list['rte_theta'] = rte_theta
        self.shp_delta = self.param_list['shp_delta'] = shp_delta
        self.rte_delta = self.param_list['rte_delta'] = rte_delta
        self.shp_phi = self.param_list['shp_phi'] = shp_phi
        self.rte_phi = self.param_list['rte_phi'] = rte_phi
        self.shp_pi = self.param_list['shp_pi'] = shp_pi
        self.rte_pi = self.param_list['rte_pi'] = rte_pi
        self.shp_delta = self.param_list['shp_delta'] = shp_delta
        self.rte_delta = self.param_list['rte_delta'] = rte_delta
        self.shp_c = self.param_list['shp_c'] = shp_c
        self.rte_c = self.param_list['rte_c'] = rte_c 
        self.lam = self.param_list['lam'] = lam
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.delta = 1.
        self.Pi_2CK = np.zeros((2, C, K))
        self.Theta_IC = np.zeros((I, C))
        self.Phi_KJ = np.zeros((K, J))
        self.Y_IC = np.zeros((I, C), dtype=int)
        self.Y_KJ = np.zeros((K, J), dtype=int)
        self.Y_2CK = np.zeros((2, C, K), dtype=int)
        self.Y_2IJ = np.zeros((2, I, J), dtype=int)
        self.Lambda_IJ = np.ones((I, J)) * self.lam
        self.c_I = np.zeros(I)
        self.c_J = np.zeros(J)

        # Cache 
        self.Mu_2IJ = np.zeros((2, I, J))

        # Auxiliary data structures
        self._P_TC = np.zeros((n_threads, C))
        self._P_TCK = np.zeros((n_threads, C, K))
        self._N_TC = np.zeros((n_threads, C), dtype=np.uint32)
        self._N_TCK = np.zeros((n_threads, C, K), dtype=np.uint32)
        self._Y_T2CK = np.zeros((n_threads, 2, C, K), dtype=int)
        self._Y_TKJ = np.zeros((n_threads, K, J), dtype=int)
        
        if self.phi_prior == 'dir_K':
            self._shp_JK = np.zeros((J, K))
            self._Phi_JK = np.zeros((J, K))
        elif self.phi_prior == 'dir_J':
            self._shp_KJ = np.zeros((K, J))
        elif (self.phi_prior == 'gamma') or (self.pi_prior == 'gamma'):
            self._zeta_K = np.zeros(K)
        
        if self.theta_prior == 'dir_I':
            self._shp_CI = np.zeros((C, I))
            self._Theta_CI = np.zeros((C, I))
        elif self.theta_prior == 'dir_C':
            self._shp_IC = np.zeros((I, C))
        elif (self.theta_prior == 'gamma') or (self.pi_prior == 'gamma'):
            self._zeta_C = np.zeros(C)

        # Copy of the data which gets imputed if any are missing
        self.Beta_IJ = np.zeros((I, J))
        
        # is_obs_IJ (1 means observed, 0 means unobserved)
        self.is_obs_IJ = np.ones((I, J), np.uint32)  # default is all beta_ij are observed
        self.any_missing = 0

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('delta', self.delta, self._update_delta),
                     ('Phi_KJ', self.Phi_KJ, self._update_Phi_KJ),
                     ('Pi_2CK', self.Pi_2CK, self._update_Pi_2CK),
                     ('Theta_IC', self.Theta_IC, self._update_Theta_IC),
                     ('Y_2IJ', self.Y_2IJ, self._update_Y_2IJ),
                     ('Y_2CK', self.Y_2CK, self._update_Y_2ICKJ),
                     ('Y_IC', self.Y_IC, self._dummy_update),
                     ('Y_KJ', self.Y_KJ, self._dummy_update),
                     ('Lambda_IJ', self.Lambda_IJ, self._update_Lambda_IJ),
                     ('c_I', self.c_I, self._update_c_I),
                     ('c_J', self.c_J, self._update_c_J)]
        return variables

    cdef void _dummy_update(self, int update_mode) nogil:
        pass

    def reset_total_itns(self):
        self._total_itns = 0

    def fit(self, data_IJ, n_itns=1000, verbose=1, initialize=True, 
            schedule={}, fix_state={}, init_state={}):

        assert data_IJ.shape == (self.I, self.J)
        assert (0 <= data_IJ).all() and (data_IJ <= 1).all()
        if isinstance(data_IJ, np.ndarray):
            data_IJ = np.ma.core.MaskedArray(data_IJ, mask=None)
        assert isinstance(data_IJ, np.ma.core.MaskedArray)
        self.Beta_IJ = data_IJ.filled(fill_value=0.0).astype('float64')  # missing entries will be imputed
        self.set_mask(is_obs_IJ=~data_IJ.mask)

        if initialize:
            if verbose:
                print('\nINITIALIZING...\n')
            self._init_state()
        
        self.set_state(init_state)
        self.set_state(fix_state)

        if 'Lambda_IJ' not in fix_state.keys():
            self._update_Lambda_IJ(update_mode=self._INFER_MODE)
        
        for k in fix_state.keys():
            schedule[k] = lambda x: False

        if verbose:
            print('\nSTARTING INFERENCE...\n')

        self._update(n_itns=n_itns, verbose=int(verbose), schedule=schedule)

    def set_mask(self, is_obs_IJ=None):
        self.any_missing = 0
        if is_obs_IJ is not None:
            assert is_obs_IJ.shape == (self.I, self.J)
            if is_obs_IJ.sum() / float(is_obs_IJ.size) < 0.3:
                print('WARNING: Less than 30 percent observed entries.')
                print('REMEMBER: 1 means observed, 0 means unobserved.')
            self.is_obs_IJ = is_obs_IJ.astype(np.uint32)
            self.any_missing = int((is_obs_IJ == 0).any())

    def set_state(self, state):
        for k, v, _ in self._get_variables():
            if k in state.keys():
                state_v = state[k]
                if k == 'delta':
                    self.delta = state_v
                else:
                    assert v.shape == state_v.shape
                    for idx in np.ndindex(v.shape):
                        v[idx] = state_v[idx]
        self.update_cache()

    def update_cache(self):
        self._compose()

    cdef void _print_state(self):
        cdef:
            int num_tokens
            double sparse, fano, theta, phi, pi

        sparse = 100 * (1 - np.count_nonzero(self.Y_2IJ) / float(self.Y_2IJ.size))
        num_tokens = np.sum(self.Y_2IJ)
        mu = np.mean(np.sum(self.Mu_2IJ, axis=0))

        print('ITERATION %d: percent of zeros: %f, num_tokens: %d, mu: %f\n' % \
                      (self.total_itns, sparse, num_tokens, mu))

    cdef void _init_state(self):
        """
        Initialize internal state.
        """
        self._generate_global_state()

    def compose(self):
        """Returns the Tucker product of Theta_IC, Pi_2CK, Phi_KJ."""
        self._compose()
        return np.array(self.Mu_2IJ)

    cdef void _compose(self):
        """
        Compute the Tucker product of delta, Theta_CI, Pi_2CK, Phi_KJ.
        """
        cdef:
            double[:,:,::1] tmp_2CJ

        tmp_2CJ = np.dot(self.Pi_2CK, self.Phi_KJ)
        self.Mu_2IJ = self.delta * np.einsum('ic,xcj->xij', self.Theta_IC, tmp_2CJ)

    def generate_state(self):
        self._generate_state()

    cdef void _generate_state(self):
        """
        Generate internal state (all model parameters and latent variables).
        """
        self._generate_global_state()
        self._generate_local_state()

    cdef void _generate_global_state(self):
        """
        Generate the global (shared) model parameters.
        """
        for key, _, update_func in self._get_variables():
            if key not in ['Y_2IJ', 'Y_2CK', 'Lambda_IJ']:
                update_func(self, update_mode=self._GENERATE_MODE)
        self._compose()

    cdef void _generate_local_state(self):
        """
        Generate the local latent variables.
        """
        self._update_Y_2IJ(update_mode=self._GENERATE_MODE)
        self._update_Y_2ICKJ(update_mode=self._GENERATE_MODE)

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """

        cdef:
            int j, i, tid
            double bm, bu, shp_mij, shp_uij
        
        bm, bu = self.bm, self.bu

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                shp_mij = bm + self.Y_2IJ[0, i, j]
                shp_uij = bu + self.Y_2IJ[1, i, j]
                self.Beta_IJ[i, j] = _sample_beta(self.rngs[tid], shp_mij, shp_uij)

        self._update_Lambda_IJ(update_mode=self._GENERATE_MODE)

    def reconstruct(self, subs=()):
        raise NotImplementedError

    def likelihood(self, subs=(), missing_vals=None):
        """Calculates the Beta likelihood at given points."""
        Y_2IJ = np.array(self.Y_2IJ)

        vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]
        return st.beta.pdf(vals,
                           self.bm + Y_2IJ[0][subs], 
                           self.bu + Y_2IJ[1][subs])

    def marginal_likelihood(self, subs=(), missing_vals=None, n_mc_samples=1000):
        """Calculates the doubly non-central Beta likelihood at given points.
        
        Marginalizes over Poisson latent variables using Monte Carlo approximation.
        """
        Mu_2IJ = self.compose()
        Mu_m_ = Mu_2IJ[0][subs]
        Mu_u_ = Mu_2IJ[1][subs]
        Lam_ = np.array(self.Lambda_IJ)[subs]

        vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]

        if self.conditional == 0:
            Y_m_M_ = rn.poisson(Mu_m_, size=(n_mc_samples,) + Mu_m_.shape)
            Y_u_M_ = rn.poisson(Mu_u_, size=(n_mc_samples,) + Mu_u_.shape)

            return st.beta.pdf(vals, 
                               self.bm + Y_m_M_, 
                               self.bu + Y_u_M_).mean(axis=0)
        else:
            return cdncb_pdf(vals, self.bm, self.bu, Mu_m_, Mu_u_, Lam_)


    #def marginal_likelihood(self, subs=(), missing_vals=None, n_mc_samples=1000):
    #    """Calculates the doubly non-central Beta likelihood at given points.
    #    
    #    Marginalizes over Poisson latent variables using Monte Carlo approximation.
    #    """
    #    Mu_2IJ = self.compose()
    #    Mu_m_ = Mu_2IJ[0][subs]
    #    Mu_u_ = Mu_2IJ[1][subs]
    #    idx = (Mu_m_ < 200) & (Mu_u_ < 200)  # where to calculate marginal likelihood exactly
    #    
    #    vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]
    #    out = np.zeros_like(vals)
    #    out[idx] = dncb_pdf(vals[idx], self.bm, self.bu, Mu_m_[idx], Mu_u_[idx])
    #    out[~idx] = dncb_pdf_mc(vals[~idx], self.bm, self.bu, Mu_m_[~idx], Mu_u_[~idx], n_samples=n_mc_samples)
    #    return out

    cdef void _update_Lambda_IJ(self, int update_mode):
        cdef:
            int i, j, tid
            double b, shp_ij, rte_ij

        if self.conditional == 0:
            b = self.bm + self.bu
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                for j in range(self.J):
                    shp_ij = b + self.Y_2IJ[0, i, j] + self.Y_2IJ[1, i, j]
                    rte_ij = self.c_I[i] * self.c_J[j]
                    self.Lambda_IJ[i, j] = _sample_gamma(self.rngs[tid], shp_ij, 1./rte_ij)
        else:
            self.Lambda_IJ[:, :] = self.lam
    
    cdef void _update_c_J(self, int update_mode):
        cdef:
            int i, j, tid
            double shp_j, rte_j

        if self.conditional == 1:
            for j in prange(self.J, schedule='static', nogil=True):
                tid = self._get_thread()

                shp_j, rte_j = self.shp_c, self.rte_c

                if update_mode == self._INFER_MODE:
                    for i in range(self.I):
                        shp_j = shp_j + self.bm + self.bu + self.Y_2IJ[0, i, j] + self.Y_2IJ[1, i, j]
                        rte_j = rte_j + self.Lambda_IJ[i, j] * self.c_I[i]

                self.c_J[j] = _sample_gamma(self.rngs[tid], shp_j, 1./rte_j)
        else:
            self.c_J[:] = 1

    cdef void _update_c_I(self, int update_mode):
        cdef:
            int i, j, tid
            double shp_i, rte_i

        if self.conditional == 1:
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()

                shp_i, rte_i = self.shp_c, self.rte_c

                if update_mode == self._INFER_MODE:
                    for j in range(self.J):
                        shp_i = shp_i + self.bm + self.bu + self.Y_2IJ[0, i, j] + self.Y_2IJ[1, i, j]
                        rte_i = rte_i + self.Lambda_IJ[i, j] * self.c_J[j]

                self.c_I[i] = _sample_gamma(self.rngs[tid], shp_i, 1./rte_i)
        else:
            self.c_I[:] = 1

    cdef void _update_Y_2IJ(self, int update_mode):
        cdef:
            int j, tid, i, x, y_xij
            double bm, bu, lam_ij, beta_ij, mu_xij, beta_xij, shp_x, sca_xij, c_i, c_j

        self._compose()

        if update_mode != self._INFER_MODE:
            """TODO: This is part is no longer correct for the conditional model.
            
            However, this will only affect testing. The inference update is correct."""
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                for j in range(self.J):
                    for x in range(2):
                        mu_xij = self.Mu_2IJ[x, i, j]
                        self.Y_2IJ[x, i, j] = gsl_ran_poisson(self.rngs[tid], mu_xij)

        elif update_mode == self._INFER_MODE:
            bm, bu = self.bm, self.bu

            self.Y_2IJ[:] = 0  # don't delete! loops assume default value is zero
            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                for j in range(self.J):
                    if self.is_obs_IJ[i, j]:  # sample y_xij from Bessel if beta_ij is observed
                        lam_ij = self.Lambda_IJ[i, j]
                        beta_ij = self.Beta_IJ[i, j]
                        for x in range(2):
                            mu_xij = self.Mu_2IJ[x, i, j]
                            beta_xij = beta_ij if x == 0 else 1 - beta_ij
                            if beta_xij > 0:
                                shp_x = bm - 1 if x == 0 else bu - 1
                                c_i = self.c_I[i]
                                c_j = self.c_J[j]
                                sca_xij = 2 * sqrt(mu_xij * lam_ij * beta_xij * c_i * c_j)
                                if sca_xij > 0:
                                    y_xij = _sample_bessel(self.rngs[tid], shp_x, sca_xij)
                                    if self.debug:
                                        with gil:
                                            # assert y_xij >= 0  # comment in for debugging (uses GIL)
                                            if not (y_xij >= 0):
                                                print(shp_x, sca_xij)
                                    self.Y_2IJ[x, i, j] = y_xij

                    else:  # if beta_ij is unobserved, impute y_xij from prior
                        for x in range(2):
                            mu_xij = self.Mu_2IJ[x, i, j]
                            self.Y_2IJ[x, i, j] = gsl_ran_poisson(self.rngs[tid], mu_xij)

    cdef void _update_Y_2ICKJ(self, int update_mode):
        cdef:
            int C, K, i, j, x, c, k, y_xij, y_xicj, y_xickj, tid

        C = self.C
        K = self.K
        
        self.Y_IC[:] = 0
        self._Y_TKJ[:] = 0
        self._Y_T2CK[:] = 0

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                for x in range(2):
                    y_xij = self.Y_2IJ[x, i, j]
                    if y_xij == 0:
                        continue    

                    for c in range(C):
                        self._P_TC[tid, c] = 0
                        for k in range(K):
                            self._P_TCK[tid, c, k] = self.Pi_2CK[x, c, k] * self.Phi_KJ[k, j]
                            self._P_TC[tid, c] += self._P_TCK[tid, c, k]
                        self._P_TC[tid, c] *= self.Theta_IC[i, c]

                    gsl_ran_multinomial(self.rngs[tid], C, y_xij, &self._P_TC[tid, 0], &self._N_TC[tid, 0])

                    for c in range(C):
                        y_xicj = self._N_TC[tid, c]
                        self.Y_IC[i, c] += y_xicj
                        
                        gsl_ran_multinomial(self.rngs[tid], K, y_xicj, &self._P_TCK[tid, c, 0], &self._N_TCK[tid, c, 0])
                        
                        for k in range(K):
                            y_xickj = self._N_TCK[tid, c, k]
                            self._Y_TKJ[tid, k, j] += y_xickj
                            self._Y_T2CK[tid, x, c, k] += y_xickj

        self.Y_2CK = np.sum(self._Y_T2CK, axis=0)
        self.Y_KJ = np.sum(self._Y_TKJ, axis=0)

    cdef void _update_Phi_KJ(self, int update_mode):
        cdef:
            int k, tid, j
            double shp_kj, rte_kj

        if self.phi_prior == 'gamma':
            if update_mode == self._INFER_MODE:
                self._zeta_K = self.delta * np.dot(np.sum(self.Theta_IC, axis=0), 
                                                   np.sum(self.Pi_2CK, axis=0))

            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()

                for j in range(self.J):
                    shp_kj = self.shp_phi
                    rte_kj = self.rte_phi

                    if update_mode == self._INFER_MODE:
                        shp_kj = shp_kj + self.Y_KJ[k, j]
                        rte_kj = rte_kj + self._zeta_K[k]

                    self.Phi_KJ[k, j] = _sample_gamma(self.rngs[tid], shp_kj, 1./rte_kj)

        elif self.phi_prior == 'dir_J':
            if update_mode != self._INFER_MODE:
                self._shp_KJ[:] = self.shp_phi
            else:
                self._shp_KJ = np.add(self.shp_phi, self.Y_KJ)

            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                _sample_dirichlet(self.rngs[tid], self._shp_KJ[k], self.Phi_KJ[k])
                
                if self.debug:
                    with gil:
                        assert self.Phi_KJ[k, 0] != -1
        
        elif self.phi_prior == 'dir_K':
            
            if update_mode != self._INFER_MODE:
                self._shp_JK[:] = self.shp_phi
            else:
                self._shp_JK = np.ascontiguousarray(np.add(self.shp_phi, np.transpose(self.Y_KJ)))

            for j in prange(self.J, schedule='static', nogil=True):
                tid = self._get_thread()
                _sample_dirichlet(self.rngs[tid], self._shp_JK[j], self._Phi_JK[j])
                
                if self.debug:
                    with gil:
                        assert self._Phi_JK[j, 0] != -1
                
            self.Phi_KJ = np.ascontiguousarray(np.transpose(self._Phi_JK))

    cdef void _update_Theta_IC(self, int update_mode):
        cdef:
            int i, tid, c
            double shp_ic, rte_ic

        if self.theta_prior == 'gamma':
            if update_mode == self._INFER_MODE:
                self._zeta_C = self.delta * np.dot(np.sum(self.Pi_2CK, axis=0), 
                                                   np.sum(self.Phi_KJ, axis=1))

            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()

                for i in range(self.I):
                    shp_ic = self.shp_theta
                    rte_ic = self.rte_theta

                    if update_mode == self._INFER_MODE:
                        shp_ic = shp_ic + self.Y_IC[i, c]
                        rte_ic = rte_ic + self._zeta_C[c]

                    self.Theta_IC[i, c] = _sample_gamma(self.rngs[tid], shp_ic, 1./rte_ic)

        elif self.theta_prior == 'dir_C':
            if update_mode != self._INFER_MODE:
                self._shp_IC[:] = self.shp_theta
            else:
                self._shp_IC = np.add(self.shp_theta, self.Y_IC)

            for i in prange(self.I, schedule='static', nogil=True):
                tid = self._get_thread()
                _sample_dirichlet(self.rngs[tid], self._shp_IC[i], self.Theta_IC[i])
                
                if self.debug:
                    with gil:
                        assert self.Theta_IC[i, 0] != -1
        
        elif self.theta_prior == 'dir_I':
            
            if update_mode != self._INFER_MODE:
                self._shp_CI[:] = self.shp_theta
            else:
                self._shp_CI = np.ascontiguousarray(np.add(self.shp_theta, np.transpose(self.Y_IC)))

            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                _sample_dirichlet(self.rngs[tid], self._shp_CI[c], self._Theta_CI[c])
                
                if self.debug:
                    with gil:
                        assert self._Theta_CI[c, 0] != -1
                
            self.Theta_IC = np.ascontiguousarray(np.transpose(self._Theta_CI))

    cdef void _update_Pi_2CK(self, int update_mode):
        cdef:
            int x, c, k, tid
            double shp_xck, rte_xck, shp1_ck, shp2_ck

        if self.pi_prior == 'gamma':
            if update_mode == self._INFER_MODE:
                self._zeta_C = np.sum(self.Theta_IC, axis=0)
                self._zeta_K = np.sum(self.Phi_KJ, axis=1)

            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                for c in range(self.C):
                    for x in range(2):
                        shp_xck = self.shp_pi
                        rte_xck = self.rte_pi

                        if update_mode == self._INFER_MODE:
                            shp_xck = shp_xck + self.Y_2CK[x, c, k]
                            rte_xck = rte_xck + self.delta * self._zeta_C[c] * self._zeta_K[k]

                        self.Pi_2CK[x, c, k] = _sample_gamma(self.rngs[tid], shp_xck, 1./rte_xck)
        
        elif self.pi_prior == 'beta':
            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                
                for c in range(self.C):
                    shp1_ck = self.shp_pi
                    shp2_ck = self.shp_pi

                    if update_mode == self._INFER_MODE:
                        shp1_ck = shp1_ck + self.Y_2CK[0, c, k]
                        shp2_ck = shp2_ck + self.Y_2CK[1, c, k]
                    
                    self.Pi_2CK[0, c, k] = _sample_beta(self.rngs[tid], shp1_ck, shp2_ck)
                    self.Pi_2CK[1, c, k] = 1 - self.Pi_2CK[0, c, k]

    cdef void _update_delta(self, int update_mode):
        cdef:
            double shp, rte, 

        shp = self.shp_delta
        rte = self.rte_delta

        if update_mode == self._INFER_MODE:
            shp += np.sum(self.Y_2IJ)
            rte += np.dot(np.sum(self.Theta_IC, axis=0),
                          np.dot(np.sum(self.Pi_2CK, axis=0), np.sum(self.Phi_KJ, axis=1)))
                            
        self.delta = _sample_gamma(self.rng, shp, 1 / rte)


def initialize_DNCBTD_with_BGNMF(model, data_IJ, verbose=0, n_itns=50):
    params = model.get_params()
    C = params['C']
    K = params['K']

    print('Initializing with BGNMF...')

    if K >= C:
        bg_model = BGNMF(n_components=K, 
                         tol=1e-2, 
                         max_iter=500, 
                         verbose=int(verbose))

        bg_model.fit(data_IJ)

        A_IK = bg_model.A_IK
        B_IK = bg_model.B_IK
        H_KJ = bg_model.H_KJ

        nmf_model = NMF(n_components=C)
        A_IC = nmf_model.fit_transform(A_IK)
        A_CK = nmf_model.components_

        B_IC = nmf_model.fit_transform(B_IK)
        B_CK = nmf_model.components_

        Pi_2CK = np.stack([A_CK, B_CK])
        Theta_IC = (A_IC + B_IC) / 2.
        Phi_KJ = H_KJ

    else:
        bg_model = BGNMF(n_components=C, 
                         tol=1e-3, 
                         max_iter=20, 
                         verbose=int(verbose))

        bg_model.fit(data_IJ)

        A_JC = bg_model.A_IK
        B_JC = bg_model.B_IK
        H_CI = bg_model.H_KJ

        nmf_model = NMF(n_components=K)
        A_JK = nmf_model.fit_transform(A_JC)
        A_KC = nmf_model.components_

        B_JK = nmf_model.fit_transform(B_JC)
        B_KC = nmf_model.components_

        Pi_2CK = np.stack([A_KC.T, B_KC.T])
        Phi_KJ = ((A_JK + B_JK) / 2.).T
        Theta_IC = H_CI.T

    delta_CK = np.ones((C, K))

    if params['theta_prior'] == 'dir_C':
        Theta_I = Theta_IC.sum(axis=1)
        Theta_IC /= Theta_I[:, np.newaxis]
        assert np.allclose(1, Theta_IC.sum(axis=1))
        delta_CK *= (Theta_I.mean() / C)
    
    elif params['theta_prior'] == 'dir_I':
        Theta_C = Theta_IC.sum(axis=0)
        Theta_IC /= Theta_C
        assert np.allclose(1, Theta_IC.sum(axis=0))
        delta_CK *= Theta_C[:, np.newaxis]

    if params['phi_prior'] == 'dir_K':
        Phi_J = Phi_KJ.sum(axis=0)
        Phi_KJ /= Phi_J
        assert np.allclose(1, Phi_KJ.sum(axis=0))
        delta_CK *= (Phi_J.mean() / K)
    
    elif params['phi_prior'] == 'dir_J':
        Phi_K = Phi_KJ.sum(axis=1)
        Phi_KJ /= Phi_K[:, np.newaxis]
        assert np.allclose(1, Phi_KJ.sum(axis=1))
        delta_CK *= Phi_K
    
    if params['pi_prior'] == 'beta':
        Pi_2CK = np.clip(Pi_2CK, a_min=1e-5, a_max=1-1e-5)
        Pi_CK = Pi_2CK.sum(axis=0)
        Pi_2CK[0] /= Pi_CK
        Pi_2CK[1] = 1-Pi_2CK[0]
        assert np.allclose(1, Pi_2CK.sum(axis=0))
        delta_CK *= Pi_CK
        delta = delta_CK.mean()
    
    elif params['pi_prior'] == 'gamma':
        Pi_2CK *= delta_CK
        delta = 1.

    fix_state = {}
    fix_state['delta'] = delta
    fix_state['Pi_2CK'] = Pi_2CK
    fix_state['Theta_IC'] = Theta_IC
    fix_state['Phi_KJ'] = Phi_KJ

    model.fit(data_IJ=data_IJ,
              n_itns=n_itns,
              initialize=True,
              fix_state=fix_state,
              verbose=verbose)

    print('\n------------------\nInitialized.\n')
    model.reset_total_itns()
