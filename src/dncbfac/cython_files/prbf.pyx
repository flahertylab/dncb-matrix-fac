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
from sample cimport _sample_gamma, _sample_beta, _sample_crt, _sample_crt_lecam, _sample_dirichlet
from bessel cimport _sample as _sample_bessel
from bessel cimport _mode as _bessel_mode


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


cdef class BNPPRBF(MCMCModel):

    cdef:
        int I, J, C, K, Y_, debug, any_missing
        double bm, bu, shp_phi, shp_gam, rte_gam, shp_beta, rte_beta
        double shp_delta, rte_delta, w_, r_, Theta_
        double[::1] r_C, w_K, gam_2, beta_2, delta_2, zeta_C, zeta_K
        double[:,::1] Theta_CI, Phi_KJ, Beta_IJ, P_TC, Lambda_IJ, shp_KJ, zeta_CK, zeta_CI
        double[:,:,::1] Pi_2CK, Mu_2IJ, P_TCK
        long[:,::1] Y_TC, Y_TK, Y_CI, L_CI
        long[:,:,::1] Y_2IJ, Y_TKJ, Y_2CK, L_2CK
        long[:,:,:,::1] Y_T2CK
        unsigned int[:,::1] N_TC, mask_IJ
        unsigned int[:,:,::1] N_TCK

    def __init__(self, int I, int J, int C, int K, double bm=1., double bu=1., 
                 double shp_phi=0.1, double shp_gam=0.1, double rte_gam=0.1, 
                 double shp_beta=10, double rte_beta=10, double shp_delta=100, double rte_delta=100, 
                 int debug=0, object seed=None, object n_threads=None):

        if n_threads is None:
            n_threads = omp_get_max_threads()
            print('Max threads: %d' % n_threads)
            
        super(BNPPRBF, self).__init__(seed=seed, n_threads=n_threads)

        # Params
        self.I = self.param_list['I'] = I
        self.J = self.param_list['J'] = J
        self.C = self.param_list['C'] = C
        self.K = self.param_list['K'] = K
        self.bm = self.param_list['bm'] = bm
        self.bu = self.param_list['bu'] = bu
        self.shp_phi = self.param_list['shp_phi'] = shp_phi
        self.shp_gam = self.param_list['shp_gam'] = shp_gam
        self.rte_gam = self.param_list['rte_gam'] = rte_gam
        self.shp_beta = self.param_list['shp_beta'] = shp_beta
        self.rte_beta = self.param_list['rte_beta'] = rte_beta
        self.shp_delta = self.param_list['shp_delta'] = shp_delta
        self.rte_delta = self.param_list['rte_delta'] = rte_delta
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.gam_2 = np.zeros(2)
        self.beta_2 = np.zeros(2)
        self.delta_2 = np.zeros(2)
        self.r_C = np.zeros(C)
        self.w_K = np.zeros(K)
        self.Phi_KJ = np.zeros((K, J))
        self.Theta_CI = np.zeros((C, I))
        self.Pi_2CK = np.zeros((2, C, K))
        self.shp_KJ = np.zeros((K, J))
        self.Y_CI = np.zeros((C, I), dtype=int)
        self.Y_2CK = np.zeros((2, C, K), dtype=int)
        self.Y_2IJ = np.zeros((2, I, J), dtype=int)
        self.Lambda_IJ = np.zeros((I, J))

        # Cache 
        self.Mu_2IJ = np.zeros((2, I, J))
        self.zeta_C = np.zeros(C)
        self.zeta_K = np.zeros(K)

        # Auxiliary data structures
        self.P_TC = np.zeros((n_threads, C))
        self.P_TCK = np.zeros((n_threads, C, K))
        self.N_TC = np.zeros((n_threads, C), dtype=np.uint32)
        self.N_TCK = np.zeros((n_threads, C, K), dtype=np.uint32)
        self.Y_T2CK = np.zeros((n_threads, 2, C, K), dtype=int)
        self.Y_TKJ = np.zeros((n_threads, K, J), dtype=int)
        self.Y_TC = np.zeros((n_threads, C), dtype=int)
        self.Y_TK = np.zeros((n_threads, K), dtype=int)
        self.zeta_CI = np.zeros((C, I))
        self.zeta_CK = np.zeros((C, K))
        self.L_CI = np.zeros((C, I), dtype=int)
        self.L_2CK = np.zeros((2, C, K), dtype=int)

        # Copy of the data
        self.Beta_IJ = np.zeros((I, J))
        
        # Masks (1 means observed, 0 means unobserved)
        self.mask_IJ = np.ones((I, J), np.uint32)  # default is all beta_ij are observed
        self.any_missing = 0

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('gam_r', self.gam_2[0], self._update_gam_r),
                     ('gam_w', self.gam_2[1], self._update_gam_w),
                     ('beta_r', self.beta_2[0], self._update_beta_r),
                     ('beta_w', self.beta_2[1], self._update_beta_w),
                     ('r_C', self.r_C, self._update_r_C),
                     ('w_K', self.w_K, self._update_w_K),
                     ('delta_theta', self.delta_2[0], self._update_delta_theta),
                     ('delta_pi', self.delta_2[1], self._update_delta_pi),
                     ('Theta_CI', self.Theta_CI, self._update_Theta_CI),
                     ('Phi_KJ', self.Phi_KJ, self._update_Phi_KJ),
                     ('Pi_2CK', self.Pi_2CK, self._update_Pi_2CK),              
                     ('Y_2IJ', self.Y_2IJ, self._update_Y_2IJ),
                     ('Y_2CK', self.Y_2CK, self._update_Y_2ICKJ),
                     ('Y_CI', self.Y_CI, self._dummy_update),
                     ('shp_KJ', self.shp_KJ, self._dummy_update),
                     ('Lambda_IJ', self.Lambda_IJ, self._update_Lambda_IJ)]
        return variables

    cdef void _dummy_update(self, int update_mode) nogil:
        pass

    def reset_total_itns(self):
        self._total_itns = 0

    def fit(self, Beta_IJ, n_itns=1000, verbose=1, initialize=True, 
            schedule={}, fix_state={}, init_state={}):

        assert Beta_IJ.shape == (self.I, self.J)
        assert (0 <= Beta_IJ).all() and (Beta_IJ <= 1).all()
        if isinstance(Beta_IJ, np.ndarray):
            Beta_IJ = np.ma.core.MaskedArray(Beta_IJ, mask=None)
        assert isinstance(Beta_IJ, np.ma.core.MaskedArray)
        self.Beta_IJ = Beta_IJ.filled(fill_value=0.0)  # missing entries are marginalized out
        self.set_mask(~Beta_IJ.mask)

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

    def set_mask(self, mask_IJ=None):
        self.any_missing = 0
        if mask_IJ is not None:
            assert mask_IJ.shape == (self.I, self.J)
            if mask_IJ.sum() / float(mask_IJ.size) < 0.3:
                print('WARNING: Less than 30 percent observed entries.')
                print('REMEMBER: 1 means observed, 0 means unobserved.')
            self.mask_IJ = mask_IJ.astype(np.uint32)
            self.any_missing = int((mask_IJ == 0).any())

    def set_state(self, state):
        for k, v, _ in self._get_variables():
            if k in state.keys():
                state_v = state[k]
                if k == 'gam_r':
                    self.gam_2[0] = state_v
                elif k == 'gam_w':
                    self.gam_2[1] = state_v
                elif k == 'beta_r':
                    self.beta_2[0] = state_v
                elif k == 'beta_w':
                    self.beta_2[1] = state_v
                elif k == 'delta_theta':
                    self.delta_2[0] = state_v
                elif k == 'delta_pi':
                    self.delta_2[1] = state_v
                else:
                    assert v.shape == state_v.shape
                    for idx in np.ndindex(v.shape):
                        v[idx] = state_v[idx]
        self.update_cache()

    def update_cache(self):
        self._tucker_prod()
        self._compute_zeta_CI()
        self._compute_zeta_CK()

    cdef void _print_state(self):
        cdef:
            int num_tokens
            double sparse, fano, theta, phi, pi

        sparse = np.count_nonzero(self.Y_2IJ) / float(self.Y_2IJ.size)
        num_tokens = np.sum(self.Y_2IJ)
        fano = (np.var(self.Y_2IJ) + 1) / (np.mean(self.Y_2IJ) + 1)
        mu = np.mean(np.sum(self.Mu_2IJ, axis=0))

        print('ITERATION %d: sparsity: %f, num_tokens: %d, fano: %f, avg mu: %f\n' % \
                      (self.total_itns, sparse, num_tokens, fano, mu))

    cdef void _init_state(self):
        """
        Initialize internal state.
        """
        self._generate_global_state()

    def tucker_prod(self):
        """Returns the Tucker product of Theta_CI, Pi_2CK, Phi_KJ."""
        return np.einsum('ci,xck,kj->xij', 
                         self.Theta_CI, 
                         self.Pi_2CK, 
                         self.Phi_KJ, 
                         optimize=True)

    cdef void _tucker_prod(self):
        """
        Compute the Tucker product of Theta_CI, Pi_2CK, Phi_KJ.
        """
        cdef:
            double[:,:,::1] tmp_2CJ

        tmp_2CJ = np.dot(self.Pi_2CK, self.Phi_KJ)
        self.Mu_2IJ = np.einsum('ci,xcj->xij', self.Theta_CI, tmp_2CJ)

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
        self._tucker_prod()

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
        Mu_2IJ = self.tucker_prod()
        Mu_m_ = Mu_2IJ[0][subs]
        Mu_u_ = Mu_2IJ[1][subs]

        Y_m_M_ = rn.poisson(Mu_m_, size=(n_mc_samples,) + Mu_m_.shape)
        Y_u_M_ = rn.poisson(Mu_u_, size=(n_mc_samples,) + Mu_u_.shape)

        vals = missing_vals if missing_vals is not None else np.array(self.Beta_IJ)[subs]
        return st.beta.pdf(vals, 
                           self.bm + Y_m_M_, 
                           self.bu + Y_u_M_).mean(axis=0)

    cpdef double log_joint_prob(self):
        """Compute the log prior probability of all latent and observed variables."""
        cdef:
            double joint_lp, shp, sca
            np.ndarray[double, ndim=3] Mu_2IJ
            np.ndarray[double, ndim=2] Beta_IJ
            np.ndarray[int, ndim=3] Y_2IJ
            np.ndarray[double, ndim=1] shp_J

        mask_IJ = np.array(self.mask_IJ).astype(bool)  # TODO: cdef this!
        Beta_IJ = np.array(self.Beta_IJ)
        Y_2IJ = np.array(self.Y_2IJ)
        joint_lp = st.beta.logpdf(Beta_IJ[mask_IJ],
                                  self.bm + Y_2IJ[0, mask_IJ],
                                  self.bu + Y_2IJ[1, mask_IJ]).sum()
        self._tucker_prod()
        Mu_2IJ = np.array(self.Mu_2IJ)
        joint_lp += st.poisson.logpmf(Y_2IJ[:, mask_IJ],
                                      Mu_2IJ[:, mask_IJ]).sum()

        shp, sca = self.gam_2[0] / self.C, 1. / self.beta_2[0]
        joint_lp += st.gamma.logpdf(self.r_C, shp, scale=sca, loc=0).sum()

        shp, sca = self.gam_2[1] / self.K, 1. / self.beta_2[1]
        joint_lp += st.gamma.logpdf(self.w_K, shp, scale=sca, loc=0).sum()

        shp, sca = self.shp_delta, 1. / self.rte_delta
        joint_lp += st.gamma.logpdf(self.delta_2, shp, scale=sca, loc=0).sum()

        shp, sca = self.shp_gam, 1. / self.rte_gam
        joint_lp += st.gamma.logpdf(self.gam_2, shp, scale=sca, loc=0).sum()

        shp, sca = self.shp_beta, 1. / self.rte_beta
        joint_lp += st.gamma.logpdf(self.beta_2, shp, scale=sca, loc=0).sum()

        shp_J = np.ones(self.J) * self.shp_phi
        joint_lp += st.dirichlet.logpdf(np.transpose(self.Phi_KJ), shp_J).sum()

        joint_lp += st.gamma.logpdf(self.Pi_2CK,
                                    self.w_K,
                                    scale=1. / self.delta_2[1],
                                    loc=0).sum()

        joint_lp += st.gamma.logpdf(np.transpose(self.Theta_CI), 
                                    self.r_C, 
                                    scale=1. / self.delta_2[0],
                                    loc=0).sum()
        return joint_lp

    cdef void _update_Lambda_IJ(self, int update_mode):
        cdef:
            int i, j, tid
            double b, shp_ij

        b = self.bm + self.bu
        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                shp_ij = b + self.Y_2IJ[0, i, j] + self.Y_2IJ[1, i, j]
                self.Lambda_IJ[i, j] = _sample_gamma(self.rngs[tid], shp_ij, 1.)

    cdef void _update_Y_2IJ(self, int update_mode):
        cdef:
            int j, tid, i, x, y_xij
            double bm, bu, lam_ij, beta_ij, mu_xij, beta_xij, shp_x, sca_xij

        self._tucker_prod()

        if update_mode != self._INFER_MODE:
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
                    if self.mask_IJ[i, j]:  # sample y_xij from Bessel if beta_ij is observed
                        lam_ij = self.Lambda_IJ[i, j]
                        beta_ij = self.Beta_IJ[i, j]
                        for x in range(2):
                            mu_xij = self.Mu_2IJ[x, i, j]
                            beta_xij = beta_ij if x == 0 else 1 - beta_ij
                            if beta_xij > 0:
                                shp_x = bm - 1 if x == 0 else bu - 1
                                sca_xij = 2 * sqrt(mu_xij * lam_ij * beta_xij)
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
        
        self.Y_CI[:] = 0
        self.Y_TKJ[:] = 0
        self.Y_T2CK[:] = 0

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            for j in range(self.J):
                for x in range(2):
                    y_xij = self.Y_2IJ[x, i, j]
                    if y_xij == 0:
                        continue    

                    for c in range(C):
                        self.P_TC[tid, c] = 0
                        for k in range(K):
                            self.P_TCK[tid, c, k] = self.Pi_2CK[x, c, k] * self.Phi_KJ[k, j]
                            self.P_TC[tid, c] += self.P_TCK[tid, c, k]
                        self.P_TC[tid, c] *= self.Theta_CI[c, i]

                    gsl_ran_multinomial(self.rngs[tid], C, y_xij, &self.P_TC[tid, 0], &self.N_TC[tid, 0])

                    for c in range(C):
                        y_xicj = self.N_TC[tid, c]
                        if y_xicj == 0:
                            continue
                        if self.mask_IJ[i, j]:
                            self.Y_CI[c, i] += y_xicj
                        
                        gsl_ran_multinomial(self.rngs[tid], K, y_xicj, &self.P_TCK[tid, c, 0], &self.N_TCK[tid, c, 0])
                        
                        for k in range(K):
                            y_xickj = self.N_TCK[tid, c, k]
                            if y_xickj == 0:
                                continue
                            self.Y_TKJ[tid, k, j] += y_xickj
                            if self.mask_IJ[i, j]:
                                self.Y_T2CK[tid, x, c, k] += y_xickj

        # TODO: Write a reduce_sources method
        self.Y_2CK = np.sum(self.Y_T2CK, axis=0, dtype=int)
        self.shp_KJ = np.add(self.shp_phi, np.sum(self.Y_TKJ, axis=0))

    cdef void _update_Phi_KJ(self, int update_mode):
        cdef:
            int k, tid

        if update_mode != self._INFER_MODE:
            self.shp_KJ[:] = self.shp_phi

        for k in prange(self.K, schedule='static', nogil=True):
            tid = self._get_thread()
            _sample_dirichlet(self.rngs[tid], self.shp_KJ[k], self.Phi_KJ[k])
            
            if self.debug:
                with gil:
                    assert self.Phi_KJ[k, 0] != -1

    cdef void _compute_zeta_CI(self):
        cdef:
            double[:,::1] Pi_CK, tmp_IK

        Pi_CK = np.sum(self.Pi_2CK, axis=0)
        tmp_IK = np.einsum('ij,kj->ik', self.mask_IJ, self.Phi_KJ)
        self.zeta_CI = np.einsum('ck,ik->ci', Pi_CK, tmp_IK)

    cdef void _update_Theta_CI(self, int update_mode):
        cdef:
            int c, tid, i
            double shp_ci, sca_ci, shp_c, sca

        if update_mode != self._INFER_MODE:
            sca = 1. / self.delta_2[0]
            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                shp_c = self.r_C[c]
                for i in range(self.I):
                    self.Theta_CI[c, i] = _sample_gamma(self.rngs[tid], shp_c, sca)          

        elif update_mode == self._INFER_MODE:
            self._compute_zeta_CI()

            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                for i in range(self.I):
                    shp_ci = self.r_C[c] + self.Y_CI[c, i]
                    sca_ci = 1. / (self.delta_2[0] + self.zeta_CI[c, i])
                    self.Theta_CI[c, i] = _sample_gamma(self.rngs[tid], shp_ci, sca_ci)          

    cdef void _update_delta_theta(self, int update_mode):
        cdef:
            double shp, sca

        if update_mode != self._INFER_MODE:
            shp = self.shp_delta
            sca = 1. / self.rte_delta

        elif update_mode == self._INFER_MODE:
            shp = self.shp_delta + self.I * np.sum(self.r_C)
            sca = 1. / (self.rte_delta + np.sum(self.Theta_CI))
        
        self.delta_2[0] = _sample_gamma(self.rng, shp, sca)

    cdef void _update_delta_pi(self, int update_mode):
        cdef:
            double shp, sca

        if update_mode != self._INFER_MODE:
            shp = self.shp_delta
            sca = 1. / self.rte_delta

        elif update_mode == self._INFER_MODE:
            shp = self.shp_delta + 2 * self.C * np.sum(self.w_K)
            sca = 1. / (self.rte_delta + np.sum(self.Pi_2CK))

        self.delta_2[1] = _sample_gamma(self.rng, shp, sca)

    cdef void _compute_zeta_CK(self):
        cdef:
            double[:,::1] tmp_IK

        tmp_IK = np.einsum('ij,kj->ik', self.mask_IJ, self.Phi_KJ)
        self.zeta_CK = np.dot(self.Theta_CI, tmp_IK)

    cdef void _update_Pi_2CK(self, int update_mode):
        cdef:
            int x, c, k, tid
            double shp_xck, sca_ck, sca, shp_k

        if update_mode != self._INFER_MODE:
            sca = 1. / self.delta_2[1]
            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                shp_k = self.w_K[k]
                for c in range(self.C):
                    for x in range(2):
                        self.Pi_2CK[x, c, k] = _sample_gamma(self.rngs[tid], shp_k, sca)

        elif update_mode == self._INFER_MODE:
            self._compute_zeta_CK()
            
            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                for k in range(self.K):
                    sca_ck = 1. / (self.delta_2[1] + self.zeta_CK[c, k])
                    for x in range(2):
                        shp_xck = self.w_K[k] + self.Y_2CK[x, c, k]
                        self.Pi_2CK[x, c, k] = _sample_gamma(self.rngs[tid], shp_xck, sca_ck)

    cdef void _update_L_CI(self):
        cdef:
            int c, tid, i, y_ci
            double r_c

        for c in prange(self.C, schedule='static', nogil=True):
            tid = self._get_thread()
            r_c = self.r_C[c]
            for i in range(self.I):
                y_ci = self.Y_CI[c, i]
                # self.L_CI[c, i] = _sample_crt(self.rngs[tid], y_ci, r_c)
                self.L_CI[c, i] = _sample_crt_lecam(self.rngs[tid], y_ci, r_c, p_min=1e-2)

    cdef void _update_r_C(self, int update_mode):
        cdef:
            int c, i, tid
            double shp_c, rte_c, shp, sca

        if update_mode != self._INFER_MODE:
            shp = self.gam_2[0] / float(self.C)
            sca = 1. / self.beta_2[0]
            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                self.r_C[c] = _sample_gamma(self.rngs[tid], shp, sca)
            self._update_L_CI()

        elif update_mode == self._INFER_MODE:
            self._update_L_CI()
            self._compute_zeta_CI()

            for c in prange(self.C, schedule='static', nogil=True):
                tid = self._get_thread()
                shp_c = self.gam_2[0] / self.C
                rte_c = self.beta_2[0]
                for i in range(self.I):
                    shp_c = shp_c + self.L_CI[c, i]
                    rte_c = rte_c + log1p(self.zeta_CI[c, i] / self.delta_2[0])
                self.r_C[c] = _sample_gamma(self.rngs[tid], shp_c, 1. / rte_c)

    cdef void _update_L_2CK(self):
        cdef:
            int c, tid, k, x, y_xck
            double w_k

        for k in prange(self.K, schedule='static', nogil=True):
            tid = self._get_thread()
            w_k = self.w_K[k]
            for c in range(self.C):
                for x in range(2):
                    y_xck = self.Y_2CK[x, c, k]
                    # self.L_2CK[x, c, k] = _sample_crt(self.rngs[tid], y_xck, w_k)
                    self.L_2CK[x, c, k] = _sample_crt_lecam(self.rngs[tid], y_xck, w_k, p_min=1e-2)

    cdef void _update_w_K(self, int update_mode):
        cdef:
            int tid, k, c, x
            double shp_k, rte_k, shp, sca

        if update_mode != self._INFER_MODE:
            shp = self.gam_2[1] / float(self.K)
            sca = 1. / self.beta_2[1]
            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                self.w_K[k] = _sample_gamma(self.rngs[tid], shp, sca)
            self._update_L_2CK()

        elif update_mode == self._INFER_MODE:
            self._update_L_2CK()
            self._compute_zeta_CK()

            for k in prange(self.K, schedule='static', nogil=True):
                tid = self._get_thread()
                shp_k = self.gam_2[1] / self.K
                rte_k = self.beta_2[1]
                for c in range(self.C):
                    rte_k = rte_k + 2 * log1p(self.zeta_CK[c, k] / self.delta_2[1])
                    for x in range(2):
                        shp_k = shp_k + self.L_2CK[x, c, k]
                self.w_K[k] = _sample_gamma(self.rngs[tid], shp_k, 1. / rte_k)

    cdef void _update_gam_r(self, int update_mode) nogil:
        cdef:
            int c, i, l_c, tid
            double gam_c, rte, shp, zeta_c
        
        if update_mode != self._INFER_MODE:
            shp, rte = self.shp_gam, self.rte_gam
            self.gam_2[0] = _sample_gamma(self.rng, shp, 1./rte)

        elif update_mode == self._INFER_MODE:
            gam_c = self.gam_2[0] / self.C

            rte = 0.
            shp = self.shp_gam
            for c in range(self.C):
                l_c = 0
                zeta_c = 0.
                for i in range(self.I):
                    l_c = l_c + self.L_CI[c, i]
                    zeta_c = zeta_c + log1p(self.zeta_CI[c, i] / self.delta_2[0])
                rte = rte + log1p(zeta_c / self.beta_2[0])
                # shp = shp + _sample_crt(self.rng, l_c, gam_c)
                shp = shp + _sample_crt_lecam(self.rng, l_c, gam_c, p_min=1e-2)
            rte = self.rte_gam + rte / self.C
            self.gam_2[0] = _sample_gamma(self.rng, shp, 1./rte)

    cdef void _update_gam_w(self, int update_mode) nogil:
        cdef:
            int k, c, x, l_k, tid
            double gam_k, rte, shp, zeta_k
        
        if update_mode != self._INFER_MODE:
            shp, rte = self.shp_gam, self.rte_gam
            self.gam_2[1] = _sample_gamma(self.rng, shp, 1./rte)
            
        elif update_mode == self._INFER_MODE:

            gam_k = self.gam_2[1] / self.K

            rte = 0.
            shp = self.shp_gam
            for k in range(self.K):
                l_k = 0
                zeta_k = 0.
                for c in range(self.C):
                    for x in range(2):
                        l_k = l_k + self.L_2CK[x, c, k]
                    zeta_k = zeta_k + 2 * log1p(self.zeta_CK[c, k] / self.delta_2[1])
                rte = rte + log1p(zeta_k / self.beta_2[1])
                # shp = shp + _sample_crt(self.rng, l_k, gam_k)
                shp = shp + _sample_crt_lecam(self.rng, l_k, gam_k, p_min=1e-2)
            rte = self.rte_gam + rte / self.K
            self.gam_2[1] = _sample_gamma(self.rng, shp, 1./rte)

    cdef void _update_beta_r(self, int update_mode):
        cdef:
            double shp, sca

        if update_mode != self._INFER_MODE:
            shp, sca = self.shp_beta, 1. / self.rte_beta

        elif update_mode == self._INFER_MODE:
            shp = self.shp_beta + self.gam_2[0]
            sca = 1. / (self.rte_beta + np.sum(self.r_C))
        
        self.beta_2[0] = _sample_gamma(self.rng, shp, sca)

    cdef void _update_beta_w(self, int update_mode):
        cdef:
            double shp, sca

        if update_mode != self._INFER_MODE:
            shp, sca = self.shp_beta, 1. / self.rte_beta
            
        elif update_mode == self._INFER_MODE:

            shp = self.shp_beta + self.gam_2[1]
            sca = 1. / (self.rte_beta + np.sum(self.w_K))
        
        self.beta_2[1] = _sample_gamma(self.rng, shp, sca)
