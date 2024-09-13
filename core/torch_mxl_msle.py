import time
import math
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim import LBFGS
from torch.quasirandom import SobolEngine


# https://github.com/pytorch/botorch/blob/main/botorch/sampling/qmc.py
class NormalQMCEngine:
    def __init__(self, d, seed = None, inv_transform = False):
        self._d = d
        self._seed = seed
        self._inv_transform = inv_transform
        if inv_transform:
            sobol_dim = d
        else:
            # to apply Box-Muller, we need an even number of dimensions
            sobol_dim = 2 * math.ceil(d / 2)
        self._sobol_engine = SobolEngine(dimension=sobol_dim, scramble=True, seed=seed)

    def draw(self, n = 1, dtype = None):
        r"""Draw `n` qMC samples from the standard Normal.

        Args:
            n: The number of samples to draw. As a best practice, use powers of 2.
            out: An option output tensor. If provided, draws are put into this
                tensor, and the function returns None.
            dtype: The desired torch data type. If None, uses `torch.get_default_dtype()`.

        Returns:
            A `n x d` tensor of samples if `out=None` and `None` otherwise.
        """
        dtype = torch.get_default_dtype() if dtype is None else dtype
        # get base samples
        samples = self._sobol_engine.draw(n, dtype=dtype)
        if self._inv_transform:
            # apply inverse transform (values to close to 0/1 result in inf values)
            v = 0.5 + (1 - torch.finfo(samples.dtype).eps) * (samples - 0.5)
            samples_tf = torch.erfinv(2 * v - 1) * math.sqrt(2)
        else:
            # apply Box-Muller transform (note: [1] indexes starting from 1)
            even = torch.arange(0, samples.shape[-1], 2)
            Rs = (-2 * torch.log(samples[:, even])).sqrt()
            thetas = 2 * math.pi * samples[:, 1 + even]
            cos = torch.cos(thetas)
            sin = torch.sin(thetas)
            samples_tf = torch.stack([Rs * cos, Rs * sin], -1).reshape(n, -1)
            # make sure we only return the number of dimension requested
            samples_tf = samples_tf[:, : self._d]

        return samples_tf


class TorchMXLMSLE(nn.Module):
    def __init__(self, dcm_dataset, batch_size, num_draws=1000, use_cuda=True, use_double=False, include_correlations=False, log_normal_params=[]):
        super(TorchMXLMSLE, self).__init__()

        self.dcm_dataset = dcm_dataset
        self.dcm_spec = dcm_dataset.dcm_spec
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.num_observations = dcm_dataset.num_observations
        self.num_alternatives = dcm_dataset.num_alternatives
        self.num_resp = dcm_dataset.num_resp
        self.num_menus = dcm_dataset.num_menus
        self.num_params = dcm_dataset.dcm_spec.num_params
        self.num_fixed_params = len(dcm_dataset.dcm_spec.fixed_param_names)
        self.num_mixed_params = len(dcm_dataset.dcm_spec.mixed_param_names)
        self.alt_attributes = dcm_dataset.alt_attributes
        self.choices = dcm_dataset.true_choices
        self.alt_availability = dcm_dataset.alt_availability
        self.mask = dcm_dataset.mask

        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")

        if use_double:
            self.use_double = True
            torch.set_default_dtype(torch.float64)
            self.torch_dtype = torch.float64
            self.numpy_dtype = np.float64
        else:
            self.use_double = False
            torch.set_default_dtype(torch.float32)
            self.torch_dtype = torch.float32
            self.numpy_dtype = np.float32

        self.num_draws = int(num_draws)
        self.seed = 1234567
        self.include_correlations = include_correlations
        self.log_normal_params = log_normal_params
        self.loglik_values = []


        # prepare data for running inference
        self.train_x = torch.tensor(self.alt_attributes, dtype=self.torch_dtype)
        self.train_y = torch.tensor(self.choices, dtype=torch.int)
        self.alt_av = torch.from_numpy(self.alt_availability)
        self.alt_av_mat = self.alt_availability.copy()
        if use_double:
            self.alt_av_mat[np.where(self.alt_av_mat == 0)] = -1e18
        else:
            self.alt_av_mat[np.where(self.alt_av_mat == 0)] = -1e9
        self.alt_av_mat -= 1

        if self.use_double:
            self.alt_av_mat_cuda = torch.from_numpy(self.alt_av_mat).double()
        else:
            self.alt_av_mat_cuda = torch.from_numpy(self.alt_av_mat).float()
        self.zeros_mat = torch.zeros(self.num_menus, self.batch_size, self.num_alternatives).to(
            self.device)  # auxiliary matrix for model
        self.alt_ids_cuda = torch.from_numpy(
            self.dcm_spec.alt_id_map[:, np.newaxis].repeat(self.num_menus * self.num_resp, 1).T.reshape(self.num_resp,
                                                                                                        self.num_menus,
                                                                                                        -1)).to(
            self.device)
        self.mask_cuda = torch.tensor(self.mask, dtype=torch.bool)

        # setup the non-linearities
        self.softplus = nn.Softplus()

        # initialize parameters
        self.initialize_parameters()


    def initialize_parameters(self, ):
        # fixed params
        alpha_mu_initial_values = torch.from_numpy(np.array(self.dcm_spec.fixed_params_initial_values, dtype=self.numpy_dtype))
        self.alpha_mu = nn.Parameter(alpha_mu_initial_values)

        if self.dcm_spec.model_type == 'MNL':
            self.zeta_mu = None
            self.zeta_cov_diag = None
            self.zeta_cov_offdiag = None
            return

        # mixed params
        zeta_mu_initial_values = torch.from_numpy(np.array(self.dcm_spec.mixed_params_initial_values, dtype=self.numpy_dtype))
        self.zeta_mu = nn.Parameter(zeta_mu_initial_values)
        self.zeta_cov_diag = nn.Parameter(torch.ones(self.num_mixed_params, dtype=self.torch_dtype))

        if self.include_correlations:
            self.zeta_cov_offdiag = nn.Parameter(
                torch.zeros(int((self.num_mixed_params * (self.num_mixed_params - 1)) / 2), dtype=self.torch_dtype))
        else:
            self.zeta_cov_offdiag = torch.zeros(int((self.num_mixed_params * (self.num_mixed_params - 1)) / 2), dtype=self.torch_dtype)

        self.tril_indices_zeta = torch.tril_indices(row=self.num_mixed_params, col=self.num_mixed_params, offset=-1)


    def loglikelihood(self, alpha_mu, zeta_mu, zeta_cov_diag, zeta_cov_offdiag, print_debug=True):

        if self.dcm_spec.model_type == 'MNL':
            beta_resp = self.gather_parameters_for_MNL_kernel(alpha_mu, None)  #, indices)
            utilities = self.compute_utilities(beta_resp, self.train_x, self.alt_av_mat_cuda, self.alt_ids_cuda)
            probs = td.Categorical(logits=utilities).log_prob(self.train_y.transpose(0, 1)).exp()  # log_prob works with mask
            probs = torch.where(self.mask_cuda.T, probs, probs.new_ones(()))  # use mask to filter out missing menus
            probs = probs.prod(axis=0)  # multiply probs over menus
            loglik_total = probs.log().sum()
        else:
            #  DEBUG - TODO: set up proper logging
            if print_debug:  # not possible for hessian comp
                print(f"before drawing - alpha = {alpha_mu.detach().cpu().numpy().tolist()}")
                print(f"before drawing - zeta = {zeta_mu.detach().cpu().numpy().tolist()}")
                print(f"before drawing - zeta_cov_diag = {zeta_cov_diag.detach().cpu().numpy().tolist()}")
                if self.include_correlations:
                    print(f"before drawing - zeta_cov_offdiag = {zeta_cov_offdiag.detach().cpu().numpy().tolist()}\n")
            # normal to draw variables from
            zeta_cov_tril = torch.zeros((self.num_mixed_params, self.num_mixed_params), dtype=self.torch_dtype, device=self.device)
            zeta_cov_tril[self.tril_indices_zeta[0], self.tril_indices_zeta[1]] = zeta_cov_offdiag
            zeta_cov_tril += torch.diag_embed(self.softplus(zeta_cov_diag))

            #torch.manual_seed(self.seed)
            #q_zeta = td.MultivariateNormal(zeta_mu, scale_tril=torch.tril(zeta_cov_tril))
            #betas = q_zeta.rsample(sample_shape=torch.Size([self.num_resp, self.num_draws]))
            betas = (self.uniform_normal_draws @ torch.tril(zeta_cov_tril) + zeta_mu)

            sampled_probs = torch.zeros(self.num_resp, device=self.device, dtype=self.torch_dtype)

            for i in range(self.num_draws):
                beta_resp = self.gather_parameters_for_MNL_kernel(alpha_mu, betas[:, i])  #, indices)
                utilities = self.compute_utilities(beta_resp, self.train_x, self.alt_av_mat_cuda, self.alt_ids_cuda)
                # TODO: maybe go from log_prob(choices).exp() to prob() and then select chosen options
                probs = td.Categorical(logits=utilities).log_prob(self.train_y.transpose(0, 1)).exp()  # log_prob works with mask
                probs = torch.where(self.mask_cuda.T, probs, probs.new_ones(()))  # use mask to filter out missing menus
                sampled_probs += probs.prod(axis=0)  # multiply probs over menus

            sampled_probs /= self.num_draws
            loglik_total = sampled_probs.log().sum()

        self.loglik_val = loglik_total.item()
        self.loglik_values.append(self.loglik_val)
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  -  loglikelihood = {self.loglik_val:.2f}")
 
        return -loglik_total


    def generate_draws(self):
        # TODO: check if we need to draw per person to keep correlation structure or if reshaping is ok here
        dist = NormalQMCEngine(self.num_mixed_params, seed=self.seed)
        self.uniform_normal_draws = dist.draw(
            self.num_draws * self.num_resp
        ).reshape([self.num_resp, self.num_draws, self.num_mixed_params]).to(device=self.device)
        assert not torch.isinf(self.uniform_normal_draws).any(),\
            f"Got infinite uniform normals for seed {self.seed}, check what is happening in engine"


    def calculate_std_errors(self):
        if self.dcm_spec.model_type == 'MNL':
            print("No std errors for MNL yet, just get rid of hstack in full_hession below")
            return None
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  -  Calculating std errors")
        #if self.dcm_spec.model_type == 'MNL':
        #    arg_nums_ = (0)
        if self.zeta_cov_offdiag.shape == torch.Size([0]):
            arg_nums_ = (0, 1, 2)
        #    ll_partial = lambda x, y, z: self.loglikelihood(x, y, z, self.zeta_cov_offdiag)
        #    hessian = torch.autograd.functional.hessian(ll_partial, (self.alpha_mu, self.zeta_mu, self.zeta_cov_diag))
        else:
            arg_nums_ = (0, 1, 2, 3)
        hess_ = torch.func.hessian(self.loglikelihood, argnums=arg_nums_)(
            self.alpha_mu, self.zeta_mu, self.zeta_cov_diag, self.zeta_cov_offdiag, False)
        full_hessian = torch.vstack([torch.hstack(x) for x in hess_])
        stderr = torch.sqrt(torch.linalg.diagonal(torch.linalg.inv(full_hessian)))
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  -  Done calculating std errors")
        return stderr.detach().cpu().numpy()


    def mask_fixed_parameters(self, fixed_params):
        """masks gradient of parameters specified in fixed_params. For random params, only the mean is masked for now."""
        fixed_param_alpha = [x for x in fixed_params if x in self.dcm_spec.fixed_param_names]
        fixed_param_zeta = [x for x in fixed_params if x in self.dcm_spec.mixed_param_names]
        # TODO: log if param is in neither

        for fixed_param in fixed_param_alpha:
            idx_var = np.where(np.array(self.dcm_spec.fixed_param_names) == fixed_param)[0]
            assert idx_var.shape[0] > 0, f"fixed var for alpha {fixed_param} not found"
            assert idx_var.shape[0] == 1, f"fixed var for alpha {fixed_param} found multiple times"
            idx_var = idx_var[0]
            self.alpha_mu.grad[idx_var] = torch.zeros(1)

        # Note only mean fixed for now
        for fixed_param in fixed_param_zeta:
            idx_var = np.where(np.array(self.dcm_spec.mixed_param_names) == fixed_param)[0]
            assert idx_var.shape[0] > 0, f"fixed var for zeta {fixed_param} not found"
            assert idx_var.shape[0] == 1, f"fixed var for zeta {fixed_param} found multiple times"
            idx_var = idx_var[0]
            self.zeta_mu.grad[idx_var] = torch.zeros(1)


    def infer(self, max_iter=50, seed=None, skip_std_err=False, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=100, fixed_params=[]):
 
        self.to(self.device)
        self.train()  # enable training mode
        self.loglik_values = []

        optimizer = LBFGS(self.parameters(), max_iter=max_iter, line_search_fn='strong_wolfe',
                          tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)

        if seed is not None:
            self.seed = int(seed)

        tic = time.time()

        if self.dcm_spec.model_type != 'MNL':
            self.generate_draws()

        def closure():
            optimizer.zero_grad()
            objective = self.loglikelihood(self.alpha_mu, self.zeta_mu, self.zeta_cov_diag, self.zeta_cov_offdiag)
            objective.backward()
            self.mask_fixed_parameters(fixed_params)
            return objective
        
        optimizer.step(closure)

        toc = time.time() - tic
        print('Elapsed time:', toc, '\n')

        # prepare python dictionary of results to output
        results = {}
        results["Estimation time"] = toc
        results["Est. alpha"] = self.alpha_mu.detach().cpu().numpy()
        results['alpha_names'] = self.dcm_spec.fixed_param_names
        if self.dcm_spec.model_type != 'MNL':
            results["Est. zeta"] = self.zeta_mu.detach().cpu().numpy()
            results['zeta_cov_diag'] = self.zeta_cov_diag.detach().cpu().numpy()
            results['zeta_cov_offdiag'] = self.zeta_cov_offdiag.detach().cpu().numpy()
            results['zeta_names'] = self.dcm_spec.mixed_param_names
        results["Loglikelihood"] = self.loglik_val
        if not skip_std_err:
            results['stderr'] = self.calculate_std_errors()
        if len(self.log_normal_params):
            results['lognormal_params'] = self.log_normal_params
        results['fixed_params'] = fixed_params
        results['loglike_values'] = self.loglik_values
        # optimizer state info
        results['optimizer_settings'] = optimizer.defaults
        k = list(optimizer.__dict__['state'].keys())[0]
        results['number_of_iterations'] = optimizer.__dict__['state'][k]['n_iter']
        results['flat_grad'] = optimizer.state[k]['prev_flat_grad'].detach().cpu().numpy()

        return results


    def compute_utilities(self, beta_resp, alt_attr, alt_avail, alt_ids):
        # compute utilities for each alternative
        utilities = torch.scatter_add(self.zeros_mat,
                                      2,
                                      alt_ids.transpose(0, 1),
                                      torch.mul(alt_attr.transpose(0, 1), beta_resp))
        # adjust utility for unavailable alternatives
        utilities += alt_avail.transpose(0, 1)
        return utilities
    

    def gather_parameters_for_MNL_kernel(self, alpha, beta):

        if self.dcm_spec.model_type == 'MNL':
            params_resp = alpha.repeat(self.batch_size, 1)
            beta_resp = torch.cat(
                [params_resp[:, self.dcm_spec.param_id_map_by_alt[i]] for i in range(self.num_alternatives)], dim=-1)
        elif self.dcm_spec.model_type == 'MXL':
            next_fixed = 0
            next_mixed = 0
            reordered_pars = []
            alpha_resp = alpha.repeat(self.batch_size, 1)
            for par_id in range(self.num_params):
                if par_id in self.dcm_spec.fixed_param_ids:
                    reordered_pars.append(alpha_resp[:, next_fixed].unsqueeze(-1))
                    next_fixed += 1
                elif par_id in self.dcm_spec.mixed_param_ids:
                    # experiment with log-normal params (assumed to be negative here)
                    if self.dcm_spec.mixed_param_names[next_mixed] in self.log_normal_params:
                        #print(f"log normal: {par_id}")
                        reordered_pars.append(-beta[:, next_mixed].unsqueeze(-1).exp())
                    else:
                        reordered_pars.append(beta[:, next_mixed].unsqueeze(-1))
                    next_mixed += 1
                else:
                    raise Exception("This should not happen - check if effect names are unique.")

            reordered_pars = torch.cat(reordered_pars, dim=-1)
            beta_resp = torch.cat(
                [reordered_pars[:, self.dcm_spec.param_id_map_by_alt[i]] for i in range(self.num_alternatives)], dim=-1)
        else:
            raise Exception("Unknown model type:", self.dcm_spec.model_type)
        
        return beta_resp
    

    def infer_bfgs(self, max_iter=None, seed=None):
        from torchmin import Minimizer
        self.to(self.device)
        self.train()  # enable training mode

        optimizer = Minimizer(self.parameters(), method='bfgs', max_iter=max_iter)

        if seed is not None:
            self.seed = int(seed)

        tic = time.time()
        self.generate_draws()

        def closure():
            optimizer.zero_grad()
            objective = self.loglikelihood(self.alpha_mu, self.zeta_mu, self.zeta_cov_diag, self.zeta_cov_offdiag)
            return objective

        optimizer.step(closure)

        toc = time.time() - tic
        print('Elapsed time:', toc, '\n')

        # prepare python dictionary of results to output
        results = {}
        results["Estimation time"] = toc
        results["Est. alpha"] = self.alpha_mu.detach().cpu().numpy()
        results['alpha_names'] = self.dcm_spec.fixed_param_names
        if self.dcm_spec.model_type != 'MNL':
            results["Est. zeta"] = self.zeta_mu.detach().cpu().numpy()
            results['zeta_cov_diag'] = self.zeta_cov_diag.detach().cpu().numpy()
            results['zeta_cov_offdiag'] = self.zeta_cov_offdiag.detach().cpu().numpy()
            results['zeta_names'] = self.dcm_spec.mixed_param_names
        results["Loglikelihood"] = self.loglik_val
        results['stderr'] = torch.sqrt(torch.linalg.diagonal(optimizer._result['hess_inv'])).detach().cpu().numpy()
        if len(self.log_normal_params):
            results['lognormal_params'] = self.log_normal_params

        return results
