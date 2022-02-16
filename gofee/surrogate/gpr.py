import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from gofee.surrogate.kernel import GaussKernel, DoubleGaussKernel
from gofee.surrogate.descriptor.fingerprint import Fingerprint
from gofee.surrogate.prior.prior import RepulsivePrior
from gofee.surrogate.gpr_calculator import gpr_calculator

class gpr_memory():
    """ Class for saving "expensive to calculate" data for
    the Gaussian Process Regression model.
    """
    def __init__(self, descriptor, prior):
        self.descriptor = descriptor
        self.prior = prior

        self.initialize_data()

    def initialize_data(self):
        self.energies = None
        self.features = None
        self.prior_values = None
        
    def get_data(self):
        return self.energies, self.features, self.prior_values
        
    def save_data(self, atoms_list, add_data=True):
        if not add_data:
            self.initialize_data()
        
        self.save_energies(atoms_list)
        self.save_features(atoms_list)
        self.save_prior_values(atoms_list)

    def save_energies(self, atoms_list):
        energies_save = np.array([a.get_potential_energy() for a in atoms_list])
        if self.energies is None:
            self.energies = energies_save
        else:
            self.energies = np.r_[self.energies, energies_save]
    
    def save_features(self, atoms_list):
        features_save = self.descriptor.get_featureMat(atoms_list)
        if self.features is None:
            self.features = features_save
        else:
            self.features = np.r_[self.features, features_save]

    def save_prior_values(self, atoms_list):
        if self.prior is not None:
            prior_values_save = np.array([self.prior.energy(a) for a in atoms_list])
            if self.prior_values is None:
                self.prior_values = prior_values_save
            else:
                self.prior_values = np.r_[self.prior_values, prior_values_save]
        else:
            self.prior_values = 0

class GPR():
    """Gaussian Process Regression
    
    Parameters:
    
    descriptor:
        Descriptor defining the represention of structures. The Gaussian Process
        works with the representations.
    
    kernel:
        Kernel (or covariance) function used in the Gaussian Process.
    
    prior:
        Prior mean function used.

    n_restarts_optimizer: int
        Number of gradient decent restarts performed by each compute process
        during hyperparameter optimization.
    """
    def __init__(self, descriptor=None, kernel='double', prior=None, n_restarts_optimizer=1, template_structure=None):
        if descriptor is None:
            self.descriptor = Fingerprint()
        else:
            self.descriptor = descriptor
        Nsplit_eta = None
        if template_structure is not None:
            self.descriptor.initialize_from_atoms(template_structure)
            if hasattr(self.descriptor, 'use_angular'):
                if self.descriptor.use_angular:
                    Nsplit_eta = self.descriptor.Nelements_2body

        if kernel is 'single':
            self.kernel = GaussKernel(Nsplit_eta=Nsplit_eta)
        elif kernel is 'double':
            self.kernel = DoubleGaussKernel(Nsplit_eta=Nsplit_eta)
        else:
            self.kernel = kernel

        if prior is None:
            self.prior = RepulsivePrior()
        else:
            self.prior = prior

        self.n_restarts_optimizer = n_restarts_optimizer

        self.memory = gpr_memory(self.descriptor, self.prior)

    def predict_energy(self, a, eval_std=False):
        """Evaluate the energy predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """
        x = self.descriptor.get_feature(a)
        k = self.kernel.kernel_vector(x, self.X)

        E = np.dot(k,self.alpha) + self.bias + self.prior.energy(a)

        if eval_std:
            # Lines 5 and 6 in GPML
            vk = np.dot(self.K_inv, k)
            E_std = np.sqrt(self.K0 - np.dot(k, vk))
            return E, E_std
        else:
            return E

    def predict_forces(self, a, eval_with_energy_std=False):
        """Evaluate the force predicted by the GPR-model.

        parameters:

        a: Atoms object
            The structure to evaluate.

        eval_with_energy_std: bool
            In addition to the force, predict also force contribution
            arrising from including the standard deviation of the
            predicted energy.
        """

        # Calculate descriptor and its gradient
        x = self.descriptor.get_feature(a)
        x_ddr = self.descriptor.get_featureGradient(a).T

        # Calculate kernel and its derivative
        k_ddx = self.kernel.kernel_jacobian(x, self.X)
        k_ddr = np.dot(k_ddx, x_ddr)

        F = -np.dot(k_ddr.T, self.alpha) + self.prior.forces(a)

        if eval_with_energy_std:
            k = self.kernel.kernel_vector(x, self.X)
            vk = np.dot(self.K_inv, k)
            g = self.K0 - np.dot(k.T, vk)
            assert g >= 0
            F_std = 1/np.sqrt(g) * np.dot(k_ddr.T, vk)
            return F.reshape((-1,3)), F_std.reshape(-1,3)
        else:
            return F.reshape(-1,3)

    def update_bias(self):
        self.bias = np.mean(self.memory.energies - self.memory.prior_values)

    def train(self, atoms_list=None, add_data=True):
        if atoms_list is not None:
            self.memory.save_data(atoms_list, add_data)

        self.update_bias()
        self.E, self.X, self.prior_values = self.memory.get_data()
        self.Y = self.E - self.prior_values - self.bias
        
        K = self.kernel(self.X)
        L = cholesky(K, lower=True)
        
        self.alpha = cho_solve((L, True), self.Y)
        self.K_inv = cho_solve((L, True), np.eye(K.shape[0]))
        self.K0 = self.kernel.kernel_value(self.X[0], self.X[0])
    
    def optimize_hyperparameters(self, atoms_list=None, add_data=True, comm=None):
        if self.n_restarts_optimizer == 0:
            self.train(atoms_list)
            return

        if atoms_list is not None:
            self.memory.save_data(atoms_list, add_data)

        self.update_bias()
        self.E, self.X, self.prior_values = self.memory.get_data()
        self.Y = self.E - self.prior_values - self.bias

        results = []
        for i in range(self.n_restarts_optimizer):
            theta_initial = np.random.uniform(self.kernel.theta_bounds[:, 0],
                                              self.kernel.theta_bounds[:, 1])
            if i == 0:
                # Make sure that the previously currently choosen
                # hyperparameters are always tried as initial values.
                if comm is not None:
                    # But only on a single communicator, if multiple are present.
                    if comm.rank == 0:
                        theta_initial = self.kernel.theta
                else:
                    theta_initial = self.kernel.theta
                        
            res = self.constrained_optimization(theta_initial)
            results.append(res)
        index_min = np.argmin(np.array([r[1] for r in results]))
        result_min = results[index_min]
        
        if comm is not None:
        # Find best hyperparameters among all communicators and broadcast.
            results_all = comm.gather(result_min, root=0)
            if comm.rank == 0:
                index_all_min = np.argmin(np.array([r[1] for r in results_all]))
                result_min = results_all[index_all_min]
            else:
                result_min = None
            result_min = comm.bcast(result_min, root=0)
                
        self.kernel.theta = result_min[0]
        self.lml = -result_min[1]

        self.train()
    
    def neg_log_marginal_likelihood(self, theta=None, eval_gradient=True):
        if theta is not None:
            self.kernel.theta = theta

        if eval_gradient:
            K, K_gradient = self.kernel(self.X, eval_gradient)
        else:
            K = self.kernel(self.X)

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.Y)

        lml = -0.5 * np.dot(self.Y, alpha)
        lml -= np.sum(np.log(np.diag(L)))
        lml -= K.shape[0]/2 * np.log(2*np.pi)
        
        if eval_gradient:
            # Equation (5.9) in GPML
            K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            tmp = np.einsum("i,j->ij", alpha, alpha) - K_inv

            lml_gradient = 0.5*np.einsum("ij,kij->k", tmp, K_gradient)
            return -lml, -lml_gradient
        else:
            return -lml

    def constrained_optimization(self, theta_initial):
        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(self.neg_log_marginal_likelihood,
                          theta_initial,
                          bounds=self.kernel.theta_bounds)
        return theta_opt, func_min

    def numerical_neg_lml(self, dx=1e-4):
        N_data = self.X.shape[0]
        theta = np.copy(self.kernel.theta)
        N_hyper = len(theta)
        lml_ddTheta = np.zeros((N_hyper))
        for i in range(N_hyper):
            theta_up = np.copy(theta)
            theta_down = np.copy(theta)
            theta_up[i] += 0.5*dx
            theta_down[i] -= 0.5*dx

            lml_up = self.neg_log_marginal_likelihood(theta_up, eval_gradient=False)
            lml_down = self.neg_log_marginal_likelihood(theta_down, eval_gradient=False)
            lml_ddTheta[i] = (lml_up - lml_down)/dx
        return lml_ddTheta

    def numerical_forces(self, a, dx=1e-4, eval_std=False):
        Na, Nd = a.positions.shape
        if not eval_std:
            F = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up = self.predict_energy(a_up)
                    E_down = self.predict_energy(a_down)
                    F[ia,idim] = -(E_up - E_down)/dx
            return F
        else:
            F = np.zeros((Na,Nd))
            Fstd = np.zeros((Na,Nd))
            for ia in range(Na):
                for idim in range(Nd):
                    a_up = a.copy()
                    a_down = a.copy()
                    a_up.positions[ia,idim] += 0.5*dx
                    a_down.positions[ia,idim] -= 0.5*dx
                    
                    E_up, Estd_up = self.predict_energy(a_up, eval_std=True)
                    E_down, Estd_down = self.predict_energy(a_down, eval_std=True)
                    F[ia,idim] = -(E_up - E_down)/dx
                    Fstd[ia,idim] = -(Estd_up - Estd_down)/dx
            return F, Fstd

    def get_calculator(self, kappa):
        return gpr_calculator(self, kappa)
    
