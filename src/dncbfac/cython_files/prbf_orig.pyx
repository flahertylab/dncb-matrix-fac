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


cdef class PRBF(MCMCModel):

    cdef:
        int I, J, C, K, Y_, debug, any_missing
        double bm, bu, shp_theta, shp_phi, shp_pi, shp_zeta, rte_zeta, zeta
        double[:,::1] Theta_IC, Phi_KJ, Beta_IJ, P_TC, Lambda_IJ, shp_KJ, shp_IC
        double[:,:,::1] Pi_2CK, shp_2CK, Mu_2IJ, P_TCK
        long[:,::1] Y_IC
        long[:,:,::1] Y_2IJ, Y_TKJ, Y_2CK
        long[:,:,:,::1] Y_T2CK
        unsigned int[:,::1] N_TC, mask_IJ
        unsigned int[:,:,::1] N_TCK

    def __init__(self, int I, int J, int C, int K, double bm=1., double bu=1., 
                 double shp_theta=0.1, double shp_pi=0.1, double shp_phi=0.1, 
                 double shp_zeta=0.1, double rte_zeta=0.1,  
                 int debug=0, object seed=None, object n_threads=None):

        if n_threads is None:
            n_threads = omp_get_max_threads()
            print('Max threads: %d' % n_threads)
            
        super(PRBF, self).__init__(seed=seed, n_threads=n_threads)

        # Params
        self.I = self.param_list['I'] = I
        self.J = self.param_list['J'] = J
        self.C = self.param_list['C'] = C
        self.K = self.param_list['K'] = K
        self.bm = self.param_list['bm'] = bm
        self.bu = self.param_list['bu'] = bu
        self.shp_theta = self.param_list['shp_theta'] = shp_theta
        self.shp_phi = self.param_list['shp_phi'] = shp_phi
        self.shp_pi = self.param_list['shp_pi'] = shp_pi
        self.shp_zeta = self.param_list['shp_zeta'] = shp_zeta
        self.rte_zeta = self.param_list['rte_zeta'] = rte_zeta
        self.debug = self.param_list['debug'] = debug

        # State variables
        self.zeta = 1.
        self.Phi_KJ = np.zeros((K, J))
        self.Theta_IC = np.zeros((I, C))
        self.Pi_2CK = np.zeros((2, C, K))
        self.shp_KJ = np.zeros((K, J))
        self.shp_IC = np.zeros((I, C))
        self.shp_2CK = np.zeros((2, C, K))
        self.Y_IC = np.zeros((I, C), dtype=int)
        self.Y_2CK = np.zeros((2, C, K), dtype=int)
        self.Y_2IJ = np.zeros((2, I, J), dtype=int)
        self.Lambda_IJ = np.zeros((I, J))

        # Cache 
        self.Mu_2IJ = np.zeros((2, I, J))

        # Auxiliary data structures
        self.P_TC = np.zeros((n_threads, C))
        self.P_TCK = np.zeros((n_threads, C, K))
        self.N_TC = np.zeros((n_threads, C), dtype=np.uint32)
        self.N_TCK = np.zeros((n_threads, C, K), dtype=np.uint32)
        self.Y_T2CK = np.zeros((n_threads, 2, C, K), dtype=int)
        self.Y_TKJ = np.zeros((n_threads, K, J), dtype=int)

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
        variables = [('zeta', self.zeta, self._update_zeta),
                     ('Theta_IC', self.Theta_IC, self._update_Theta_IC),
                     ('Phi_KJ', self.Phi_KJ, self._update_Phi_KJ),
                     ('Pi_2CK', self.Pi_2CK, self._update_Pi_2CK),              
                     ('Y_2IJ', self.Y_2IJ, self._update_Y_2IJ),
                     ('shp_2CK', self.shp_2CK, self._update_Y_2ICKJ),
                     ('shp_IC', self.shp_IC, self._dummy_update),
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
                if k == 'zeta':
                    self.zeta = state_v
                else:
                    assert v.shape == state_v.shape
                    for idx in np.ndindex(v.shape):
                        v[idx] = state_v[idx]
        self.update_cache()

    def update_cache(self):
        self._tucker_prod()

    cdef void _print_state(self):
        cdef:
            int num_tokens
            double sparse, fano, theta, phi, pi

        sparse = 100 * (1 - np.count_nonzero(self.Y_2IJ) / float(self.Y_2IJ.size))
        num_tokens = np.sum(self.Y_2IJ)
        mu = np.mean(np.sum(self.Mu_2IJ, axis=0))

        print('ITERATION %d: percent of zeros: %f, num_tokens: %d, zeta: %f\n' % \
                      (self.total_itns, sparse, num_tokens, self.zeta))

    cdef void _init_state(self):
        """
        Initialize internal state.
        """
        self._generate_global_state()

    def tucker_prod(self):
        """Returns the Tucker product of Theta_IC, Pi_2CK, Phi_KJ."""
        self._tucker_prod()
        return np.array(self.Mu_2IJ)

    # cdef void _tucker_prod(self):
    #     """
    #     Compute the Tucker product of Theta_IC, Pi_2CK, Phi_KJ.
    #     """
    #     cdef:
    #         double[:,:,::1] tmp_2CJ

    #     self.Mu_2IJ = np.einsum('ic,xck,kj->xij',
    #                             self.Theta_IC, 
    #                             self.Pi_2CK, 
    #                             self.Phi_KJ, 
    #                             optimize=True) * self.zeta

    cdef void _tucker_prod(self):
        """
        Compute the Tucker product of Theta_CI, Pi_2CK, Phi_KJ.
        """
        cdef:
            double[:,:,::1] tmp_2CJ

        tmp_2CJ = np.dot(self.Pi_2CK, self.Phi_KJ)
        self.Mu_2IJ = self.zeta * np.einsum('ic,xcj->xij', self.Theta_IC, tmp_2CJ)

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

        shp_J = np.ones(self.J) * self.shp_phi
        joint_lp += st.dirichlet.logpdf(np.transpose(self.Phi_KJ), shp_J).sum()

        raise NotImplementedError
        # joint_lp += st.beta.logpdf(self.Pi_2CK[0],
        #                            None,
        #                            None,
        #                            loc=0).sum()

        # joint_lp += st.gamma.logpdf(np.transpose(self.Theta_IC), 
        #                             self.r_C, 
        #                             scale=1. / self.delta_2[0],
        #                             loc=0).sum()
        # return joint_lp

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
        
        self.Y_IC[:] = 0
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
                        self.P_TC[tid, c] *= self.Theta_IC[i, c]

                    gsl_ran_multinomial(self.rngs[tid], C, y_xij, &self.P_TC[tid, 0], &self.N_TC[tid, 0])

                    for c in range(C):
                        y_xicj = self.N_TC[tid, c]
                        self.Y_IC[i, c] += y_xicj
                        
                        gsl_ran_multinomial(self.rngs[tid], K, y_xicj, &self.P_TCK[tid, c, 0], &self.N_TCK[tid, c, 0])
                        
                        for k in range(K):
                            y_xickj = self.N_TCK[tid, c, k]
                            self.Y_TKJ[tid, k, j] += y_xickj
                            self.Y_T2CK[tid, x, c, k] += y_xickj

        # TODO: Write a reduce_sources method
        self.shp_2CK = np.add(self.shp_pi, np.sum(self.Y_T2CK, axis=0))
        self.shp_KJ = np.add(self.shp_phi, np.sum(self.Y_TKJ, axis=0))
        self.shp_IC = np.add(self.shp_theta, self.Y_IC)

    cdef void _update_zeta(self, int update_mode):
        cdef:
            double shp, rte

        shp = self.shp_zeta
        rte = self.rte_zeta

        if update_mode == self._INFER_MODE:
            shp += np.sum(self.Y_2IJ)
            rte += self.I * self.K

        self.zeta = _sample_gamma(self.rng, shp, 1 / rte)

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

    cdef void _update_Theta_IC(self, int update_mode):
        cdef:
            int i, tid

        if update_mode != self._INFER_MODE:
            self.shp_IC[:] = self.shp_theta

        for i in prange(self.I, schedule='static', nogil=True):
            tid = self._get_thread()
            _sample_dirichlet(self.rngs[tid], self.shp_IC[i], self.Theta_IC[i])
            
            if self.debug:
                with gil:
                    assert self.Theta_IC[i, 0] != -1        

    cdef void _update_Pi_2CK(self, int update_mode):
        cdef:
            int c, k, tid
            double shp1_ck, shp2_ck

        if update_mode != self._INFER_MODE:
            self.shp_2CK[:] = self.shp_pi

        for c in prange(self.C, schedule='static', nogil=True):
            tid = self._get_thread()
            
            for k in range(self.K):
                shp1_ck = self.shp_2CK[0, c, k]
                shp2_ck = self.shp_2CK[1, c, k]
                self.Pi_2CK[0, c, k] = _sample_beta(self.rngs[tid], shp1_ck, shp2_ck)
                self.Pi_2CK[1, c, k] = 1 - self.Pi_2CK[0, c, k]

