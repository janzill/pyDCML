import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim import Adam
import numpy as np
import time


class TorchMXLMSLE(nn.Module):
    def __init__(self, dcm_dataset, batch_size, num_draws=1000, use_cuda=True, use_double=False):
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
        self.context = dcm_dataset.context

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

        self.num_draws = num_draws

        # prepare data for running inference
        self.train_x = torch.tensor(self.alt_attributes, dtype=self.torch_dtype)  # .to(self.device)
        self.context_info = torch.tensor(self.context, dtype=self.torch_dtype)  # .to(self.device)
        self.train_y = torch.tensor(self.choices, dtype=torch.int)  # .to(self.device)
        self.alt_av = torch.from_numpy(self.alt_availability)  # .to(self.device)
        self.alt_av_mat = self.alt_availability.copy()
        self.alt_av_mat[np.where(self.alt_av_mat == 0)] = -1e9
        self.alt_av_mat -= 1

        if self.use_double:
            self.alt_av_mat_cuda = torch.from_numpy(self.alt_av_mat).double()  # .to(self.device)
        else:
            self.alt_av_mat_cuda = torch.from_numpy(self.alt_av_mat).float()  # .to(self.device)
        self.zeros_mat = torch.zeros(self.num_menus, self.batch_size, self.num_alternatives).to(
            self.device)  # auxiliary matrix for model
        self.alt_ids_cuda = torch.from_numpy(
            self.dcm_spec.alt_id_map[:, np.newaxis].repeat(self.num_menus * self.num_resp, 1).T.reshape(self.num_resp,
                                                                                                        self.num_menus,
                                                                                                        -1)).to(
            self.device)
        self.mask_cuda = torch.tensor(self.mask, dtype=torch.bool)  # .to(self.device)

        # setup the non-linearities
        self.softplus = nn.Softplus()

        # initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self, ):
        # fixed params
        alpha_mu_initial_values = torch.from_numpy(np.array(self.dcm_spec.fixed_params_initial_values, dtype=self.numpy_dtype))
        self.alpha_mu = nn.Parameter(alpha_mu_initial_values)
        # self.alpha_cov_diag = nn.Parameter(torch.ones(self.num_fixed_params))
        # self.alpha_cov_offdiag = nn.Parameter(
        #    torch.zeros(int((self.num_fixed_params * (self.num_fixed_params - 1)) / 2)))
        # self.tril_indices_alpha = torch.tril_indices(row=self.num_fixed_params, col=self.num_fixed_params, offset=-1)

        # mixed params
        zeta_mu_initial_values = torch.from_numpy(np.array(self.dcm_spec.mixed_params_initial_values, dtype=self.numpy_dtype))
        self.zeta_mu = nn.Parameter(zeta_mu_initial_values)
        self.zeta_cov_diag = nn.Parameter(torch.ones(self.num_mixed_params))
        # NO CORRELATIONS INITIALLY
        self.zeta_cov_offdiag = torch.zeros(int((self.num_mixed_params * (self.num_mixed_params - 1)) / 2))
        # self.zeta_cov_offdiag = nn.Parameter(
        #     torch.zeros(int((self.num_mixed_params * (self.num_mixed_params - 1)) / 2)))
        self.tril_indices_zeta = torch.tril_indices(row=self.num_mixed_params, col=self.num_mixed_params, offset=-1)


    def loglik(self, alt_attr, context_attr, obs_choices, alt_avail, obs_mask, alt_ids, indices):

        # normal to draw variables from
        zeta_cov_tril = torch.zeros((self.num_mixed_params, self.num_mixed_params), device=self.device)
        zeta_cov_tril[self.tril_indices_zeta[0], self.tril_indices_zeta[1]] = self.zeta_cov_offdiag
        zeta_cov_tril += torch.diag_embed(self.softplus(self.zeta_cov_diag))
        q_zeta = td.MultivariateNormal(self.zeta_mu, scale_tril=torch.tril(zeta_cov_tril))

        # todo: add draws as dimensions and map over for efficiency
        loglik_total = 0.0
        for i in range(self.num_draws):
            # draw num_person times from this to create individual params
            beta = q_zeta.rsample(sample_shape=torch.Size([self.num_resp]))
            # ----- gather paramters for computing the utilities -----
            beta_resp = self.gather_parameters_for_MNL_kernel(self.alpha_mu, beta, indices)
            # ----- compute utilities -----
            utilities = self.compute_utilities(beta_resp, alt_attr, alt_avail, alt_ids)
            # ----- (expected) log-likelihood -----
            loglik = td.Categorical(logits=utilities).log_prob(obs_choices.transpose(0, 1))
            loglik = torch.where(obs_mask.T, loglik, loglik.new_zeros(()))  # use mask to filter out missing menus
            loglik = loglik.sum()
            loglik_total += loglik

        loglik_total /= self.num_draws
        # compute accuracy based on utilities
        # acc = utilities.argmax(-1) == obs_choices.transpose(0, 1)
        # acc = torch.where(obs_mask.T, acc, acc.new_zeros(()))
        # acc = acc.sum() / obs_mask.sum()

        return - loglik_total


    def infer(self, num_epochs=10000, learning_rate=1e-2, return_all_results=False):
 
        self.to(self.device)

        optimizer = Adam(self.parameters(), lr=learning_rate)

        self.train()  # enable training mode

        tic = time.time()

        all_results = [] # TODO: quick hack for looking at results over epochs, set up proper monitoring

        for epoch in range(num_epochs):
            permutation = torch.randperm(self.num_resp)

            for i in range(0, self.num_resp, self.batch_size):

                indices = permutation[i:i + self.batch_size]
                batch_x, batch_context, batch_y = self.train_x[indices], self.context_info[indices], self.train_y[
                    indices]
                batch_alt_av_mat, batch_mask_cuda, batch_alt_ids = self.alt_av_mat_cuda[indices], self.mask_cuda[
                    indices], self.alt_ids_cuda[indices]

                batch_x = batch_x.to(self.device)
                batch_context = batch_context.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_alt_av_mat = batch_alt_av_mat.to(self.device)
                batch_mask_cuda = batch_mask_cuda.to(self.device)

                optimizer.zero_grad()
                loglik = self.loglik(batch_x, batch_context, batch_y, batch_alt_av_mat, batch_mask_cuda, batch_alt_ids,
                                 indices)
                loglik.backward()
                optimizer.step()

                if not epoch % 10:
                    print("[Epoch %5d] Loglik: %.1f" % (epoch, loglik.item()))

                if return_all_results:
                    results_this_epoch = {}
                    results_this_epoch["alpha_mu"] = self.alpha_mu.detach().cpu().numpy().tolist()
                    results_this_epoch["zeta_mu"] = self.zeta_mu.detach().cpu().numpy().tolist()
                    results_this_epoch['zeta_cov_diag'] = self.zeta_cov_diag.detach().cpu().numpy().tolist()
                    results_this_epoch['zeta_cov_offdiag'] = self.zeta_cov_offdiag.detach().cpu().numpy().tolist()
                    results_this_epoch["Loglikelihood"] = loglik.item()
                    all_results.append(results_this_epoch)

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
        results["Loglikelihood"] = loglik.item()
        results['num_epochs'] = num_epochs

        # show quick summary of results
        print(f"Loglik at end of training = {loglik.item():.1f}")
        # print("Est. alpha:", self.alpha_mu.detach().cpu().numpy())
        # for i in range(len(self.dcm_spec.fixed_param_names)):
        #     print("\t%s: %.3f" % (self.dcm_spec.fixed_param_names[i], results["Est. alpha"][i]))
        # print()
        # if self.dcm_spec.model_type != 'MNL':
        #     print("Est. zeta:", self.zeta_mu.detach().cpu().numpy())
        #     for i in range(len(self.dcm_spec.mixed_param_names)):
        #         print("\t%s: %.3f" % (self.dcm_spec.mixed_param_names[i], results["Est. zeta"][i]))
        #     print()

        return results, all_results


    def compute_utilities(self, beta_resp, alt_attr, alt_avail, alt_ids):
        # compute utilities for each alternative
        utilities = torch.scatter_add(self.zeros_mat,
                                      2,
                                      alt_ids.transpose(0, 1),
                                      torch.mul(alt_attr.transpose(0, 1), beta_resp))
        # adjust utility for unavailable alternatives
        utilities += alt_avail.transpose(0, 1)
        return utilities
    

    def gather_parameters_for_MNL_kernel(self, alpha, beta, indices):

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
                    reordered_pars.append(beta[indices, next_mixed].unsqueeze(-1))
                    next_mixed += 1
                else:
                    raise Exception("This should not happen - check if effect names are unique.")

            reordered_pars = torch.cat(reordered_pars, dim=-1)
            beta_resp = torch.cat(
                [reordered_pars[:, self.dcm_spec.param_id_map_by_alt[i]] for i in range(self.num_alternatives)], dim=-1)
        else:
            raise Exception("Unknown model type:", self.dcm_spec.model_type)
        
        return beta_resp