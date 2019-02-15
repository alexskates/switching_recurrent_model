import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils import (
    init_weights,
    gumbel_softmax,
    loglik_mixture,
    offset,
    reparam_normal,
    gaussian_kl_div
)
from modules import (
    TCNDistribution,
    RNNDistribution,
    RNNSoftmax,
    TCNSoftmax,
    MarkovSoftmax,
    Uninform
)

_eps = 1e-20


# Straightforward Categorical distribution for the state probabilities (
# approximated by the Gumbel Softmax)
class GumbelEncoder(nn.Module):
    """
    The encoder module contains the "inference" network, that maps the data
    to the distribution over z. This module is simply a wrapper which decides
    which network we use for inference.

    Note that no sampling actually takes place inside this module,
    that is all done in the decoder, which contains the generative model.

    """
    def __init__(self, params):
        super(GumbelEncoder, self).__init__()

        # For purposes of saving the results
        self.distributions = {'z': params.n_latent}

        if params.inference.lower() == 'tcn':
            self.inference = TCNDistribution(params)
        else:
            self.inference = RNNDistribution(params)

    def forward(self, x, h=None):
        _, logits_q_z, h = self.inference((x, h))
        # Actually softmax the logits to get the "real distribution" that the
        #  gumbel softmax is an approximate sample from
        q_z = F.softmax(logits_q_z, dim=-1)
        return logits_q_z, {'z': q_z}, h

    def init_hidden(self, batch_size):
        if type(self.inference) == RNNDistribution:
            return self.inference.init_hidden(batch_size)
        else:
            return None


class GumbelDecoder(nn.Module):
    def __init__(self, params):
        super(GumbelDecoder, self).__init__()

        # Generative model
        if params.prior.lower() == 'rnn':
            self.prior = RNNSoftmax(params)
        elif params.prior.lower() == 'tcn':
            self.prior = TCNSoftmax(params)
        elif params.prior.lower() == 'markov':
            self.prior = MarkovSoftmax(params)
        else:
            self.prior = Uninform(params)

        self.B = nn.Parameter(torch.from_numpy(params.B_init))
        self.W = nn.Parameter(torch.from_numpy(params.W_init))

        if params.cuda:
            self.cuda()

    def forward(self, tau, logits_z, h=None):
        z = gumbel_softmax(logits_z, tau, hard=False)

        # p(z_t|z_{:t-1})
        z_offset = offset(z)
        # If we need to pass in a hidden state to the generative model,
        # the class will have an attribute specifying how big it it
        _, p_z, h = self.prior((z_offset, h))

        # Covariances
        c = self.covariance()
        # Ensure is positive definite
        c += torch.eye(c.shape[-1], dtype=z.dtype, device=z.data.device) * 1e-4
        return c, {'z': z}, {'z': p_z}, h

    def covariance(self):
        cov = self.W.matmul(self.B)
        cov = cov.matmul(torch.transpose(cov, -1, -2))
        return cov

    def init_hidden(self, batch_size):
        if type(self.prior) == RNNSoftmax:
            return self.prior.init_hidden(batch_size)
        else:
            return None


def gumbel_layer(n_in, n_hidden, n_out, bidirectional, *args, **kwargs):
    n_directions = 2 if bidirectional else 1
    return nn.ModuleList([
        nn.GRU(
            input_size=n_in,
            hidden_size=n_hidden,
            bidirectional=bidirectional,
            batch_first=True,
            *args,
            **kwargs
        ),
        nn.Linear(
            in_features=n_hidden * n_directions,
            out_features=n_out
        )
    ])


class FactoredHierarchicalEncoder(nn.Module):
    """
    The encoder module contains the "inference" network, that maps the data
    to the distribution over z. This module is simply a wrapper which decides
    which network we use for inference.

    We must sample from the normal distribution to get the logits for the
    gumbel distribution

    """
    def __init__(self, params):
        super(FactoredHierarchicalEncoder, self).__init__()

        # For purposes of saving the results
        self.distributions = {'z': params.n_latent}

        self.num_directions = 2 if params.bidirectional else 1

        # params.n_latent is a list of the number of states in each layer.
        # For the variational distribution, we need to reverse this
        self.input_sizes = params.n_input
        self.output_sizes = sum(params.n_latent)
        self.latent_dims = list(reversed(params.n_latent))
        self.n_hidden = params.n_hidden[0]

        self.inference = gumbel_layer(self.input_sizes, self.n_hidden,
                                      self.output_sizes, params.bidirectional)

        # self.linear_layers = nn.ModuleList([
        #     nn.Linear(self.output_sizes, latent_dim) for latent_dim in
        #     self.latent_dims
        # ])

        # Weight initialization
        self.inference.apply(init_weights)
        # self.linear_layers.apply(init_weights)

        if params.cuda:
            self.cuda()

    def forward(self, x, h=None):
        q_layers = []; q_samples = []

        # Fully factored variational distribution, where all layers are being
        # output but the gru at the same time, but concatenated
        out, h = self.inference[0](x, h)
        out = self.inference[1](out)

        idx = np.insert(np.cumsum(self.latent_dims), 0, 0)

        for idx_start, idx_end in zip(idx[:-1], idx[1:]):
            layer_logits = out[:, :, idx_start:idx_end]

            # Softmax to get the "true" variational distribution, GS to get a
            # differentiable approximation to the variational distribution
            q_layers.append(F.softmax(layer_logits, dim=-1))
            q_samples.append(gumbel_softmax(layer_logits, 0.5, hard=False))

        return q_samples, {'z': q_layers}, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.num_directions, batch_size,
                                   self.n_hidden).zero_())


class HierarchicalEncoder(nn.Module):
    """
    The encoder module contains the "inference" network, that maps the data
    to the distribution over z. This module is simply a wrapper which decides
    which network we use for inference.

    We must sample from the normal distribution to get the logits for the
    gumbel distribution

    """
    def __init__(self, params):
        super(HierarchicalEncoder, self).__init__()

        # For purposes of saving the results
        self.distributions = {'z': params.n_latent}

        self.num_directions = 2 if params.bidirectional else 1

        # params.n_latent is a list of the number of states in each layer.
        # For the variational distribution, we need to reverse this
        self.input_sizes = [params.n_input, *list(reversed(params.n_latent[1:]))]
        self.output_sizes = list(reversed(params.n_latent))
        self.n_hidden = params.n_hidden

        inference_blocks = [
            gumbel_layer(n_in, n_hid, n_out, params.bidirectional)
            for n_in, n_hid, n_out in zip(self.input_sizes, self.n_hidden,
                                          self.output_sizes)
        ]

        self.inference_layers = nn.ModuleList(inference_blocks)

        # Weight initialization
        self.inference_layers.apply(init_weights)

        if params.cuda:
            self.cuda()

    def forward(self, x, h=None):
        q_layers = []; q_samples = []; h_new = []

        in_ = x  # Set in_ to x for the first layer

        # Each element of self.inference_layers is a ModuleList
        # corresponding to a hierarchical layer
        for i, layer in enumerate(self.inference_layers):
            # Each inference layer "l" is a ModuleList with a GRU followed by
            # linear layer
            out, h_layer = layer[0](in_, h[i])
            logits = layer[1](out)
            h_new.append(h_layer)

            # Softmax to get the "true" variational distribution, GS to get a
            # differentiable approximation to the variational distribution
            q_layers.append(F.softmax(logits, dim=-1))
            q_samples.append(gumbel_softmax(logits, 0.5, hard=False))

            in_ = q_layers[i]  # Each layer is conditioned on the previous

        return q_samples, {'z': q_layers}, h_new

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return [Variable(weight.new(self.num_directions, batch_size, n_hid).zero_())
                for n_hid in self.n_hidden]


class HierarchicalDecoder(nn.Module):
    def __init__(self, params):
        super(HierarchicalDecoder, self).__init__()

        # A set of transition matrices for each level of the hierarchy
        # Each layer d has K^{d}
        top_layer = nn.Parameter(
            torch.ones(params.n_latent[0], params.n_latent[0]) +
            torch.eye(params.n_latent[0]) * params.diag)
        lower_layers = [
            nn.Parameter(torch.ones(*sz) + torch.eye(sz[-1]) * params.diag)
            for sz in zip(params.n_latent[:-1], params.n_latent[1:],
                          params.n_latent[1:])
        ]
        self.prior_parameters = nn.ParameterList([top_layer, *lower_layers])

        self.B = nn.Parameter(torch.from_numpy(params.B_init))
        self.W = nn.Parameter(torch.from_numpy(params.W_init))

        if params.cuda:
            self.cuda()

    def forward(self, tau, z_samples, h=None):

        # Do the "top" layer -- just simple matrix multiplication
        z_current_layer = offset(z_samples[-1])
        p_z = [z_current_layer.matmul(self.trans_prob()[0])]

        for i, trans_prob in enumerate(self.trans_prob()[1:]):
            z_previous_layer = z_current_layer
            z_current_layer = offset(z_samples[-(2+i)])

            # The actual trans prob to use in this layer is indexed by the
            # variable in the layer above this one. As the variables are
            # one-hot, you can achieve indexing with matrix multiplication.
            # The variable in this layer indexes which row to use. Shorthand
            # of all these operations can be achieved with torch.einsum. See
            # https://pytorch.org/docs/stable/torch.html#torch.einsum for
            # details to save explaining it all here

            # b: batch, l: length, i: number of states of previous layer
            # j: number of states in current layer, k = j
            p_z_layer = torch.einsum('bli,blj,ijk->blk', z_previous_layer,
                                     z_current_layer, trans_prob)
            p_z.append(p_z_layer)

        # Covariances
        c = self.covariance()
        # Ensure is positive definite
        c += torch.eye(c.shape[-1], dtype=z_current_layer.dtype,
                       device=z_current_layer.data.device) * 1e-4

        return c, {'z': z_samples}, {'z': p_z}, h

    def covariance(self):
        cov = self.W.matmul(self.B)
        cov = cov.matmul(torch.transpose(cov, -1, -2))
        return cov

    def trans_prob(self):
        return [F.softmax(p_matrix, dim=-1) for p_matrix in
                self.prior_parameters]

    def init_hidden(self, batch_size):
        return None


# Normal distribution over the logits of the Categorical/Gumbel distribution
class GumbelNormalEncoder(nn.Module):
    """
    The encoder module contains the "inference" network, that maps the data
    to the distribution over z. This module is simply a wrapper which decides
    which network we use for inference.

    We must sample from the normal distribution to get the logits for the
    gumbel distribution

    """
    def __init__(self, params):
        super(GumbelNormalEncoder, self).__init__()

        # For purposes of saving the results
        self.distributions = {'alpha': params.n_latent,
                              'z': params.n_latent}

        if params.inference.lower() == 'tcn':
            self.inference = TCNDistribution(params)
        else:
            self.inference = RNNDistribution(params)

        self.lv = nn.Parameter(torch.tensor([-2.], dtype=torch.float32))

        if params.cuda:
            self.cuda()

    def forward(self, x, h=None):
        # Introduce extra stochastic layer compared to gumbel
        _, q_alpha_mu, h = self.inference((x, h))
        q_alpha_lv = self.lv

        # Sample from normal distribution
        alpha = reparam_normal(q_alpha_mu, q_alpha_lv, self.training)

        # Actually softmax the logits to get the "real distribution" that the
        #  gumbel softmax is an approximate sample from
        q_z = F.softmax(alpha, dim=-1)

        return alpha, {'alpha': (q_alpha_mu, q_alpha_lv),
                       'z': q_z}, h

    def init_hidden(self, batch_size):
        if type(self.inference) == RNNDistribution:
            return self.inference.init_hidden(batch_size)
        else:
            return None


class GumbelNormalDecoder(nn.Module):
    def __init__(self, params):
        super(GumbelNormalDecoder, self).__init__()

        # Generative model
        if params.prior.lower() == 'rnn':
            params.bidirectional = False
            self.prior = RNNDistribution(params)
        elif params.prior.lower() == 'tcn':
            self.prior = TCNDistribution(params)

        self.B = nn.Parameter(torch.from_numpy(params.B_init))
        self.W = nn.Parameter(torch.from_numpy(params.W_init))

        self.lv = nn.Parameter(torch.tensor([-2.], dtype=torch.float32))

        if params.cuda:
            self.cuda()

    def forward(self, tau, alpha, h=None):
        # Unlike the gumbel version, the prior is over alpha, not z.
        # p(alpha_t|alpha_{:t-1})
        alpha_offset = offset(alpha)
        _, p_alpha_mu, h = self.prior((alpha_offset, h))
        p_alpha_lv = self.lv

        # Sample from alpha to calculate p_z
        alpha_0 = reparam_normal(p_alpha_mu, p_alpha_lv, self.training)

        p_z = F.softmax(alpha_0, dim=-1)

        # Sample from approximation of q_z for MC Expectation of NLL
        z = gumbel_softmax(alpha, tau, hard=False)


        # Covariances
        c = self.covariance()
        # Ensure is positive definite
        c += torch.eye(c.shape[-1], dtype=z.dtype, device=z.data.device) * 1e-4
        return c, \
               {'alpha': alpha, 'z': z}, \
               {'alpha': (p_alpha_mu, p_alpha_lv), 'z': p_z}, \
               h

    def covariance(self):
        cov = self.W.matmul(self.B)
        cov = cov.matmul(torch.transpose(cov, -1, -2))
        return cov

    def init_hidden(self, batch_size):
        if type(self.prior) == RNNSoftmax:
            return self.prior.init_hidden(batch_size)
        else:
            return None


def hierarchical_vfe(x, expected_values, covariances, variational_distribution,
                     prior_distribution):
    # Log likelihood
    ll = loglik_mixture(x.contiguous(),
                        expected_values['z'][0].contiguous(),
                        covariances)

    vfe = -ll

    # KL divergence for each hierarchical layer
    for i in range(len(prior_distribution['z'])):
        q_z = variational_distribution['z'][-(i+1)].contiguous()
        p_z = prior_distribution['z'][i].contiguous()
        log_q_z = torch.log(q_z + _eps)
        log_p_z = torch.log(p_z + _eps)
        kl_layer = torch.sum(q_z * (log_q_z - log_p_z), -1)
        vfe = vfe + kl_layer

    return vfe


def gumbel_vfe(x, expected_values, covariances, variational_distribution,
               prior_distribution):
    # KL divergence
    q_z = variational_distribution['z'].contiguous()
    p_z = prior_distribution['z'].contiguous()
    log_q_z = torch.log(q_z + _eps)
    log_p_z = torch.log(p_z + _eps)
    kl = torch.sum(q_z * (log_q_z - log_p_z), -1)

    # Log likelihood
    ll = loglik_mixture(x.contiguous(),
                        expected_values['z'].contiguous(),
                        covariances)

    elbo = ll - kl  # ELBO / variational free energy
    return -elbo


def gumbel_normal_vfe(x, expected_values, covariances, variational_distribution,
                      prior_distribution):
    # Negative log likelihood
    ll = loglik_mixture(x.contiguous(),
                        expected_values['z'].contiguous(),
                        covariances)

    # KL Divergence
    q_z = variational_distribution['z'].contiguous()
    p_z = prior_distribution['z'].contiguous()
    log_q_z = torch.log(q_z + _eps)
    log_p_z = torch.log(p_z + _eps)
    kl_z = torch.sum(q_z * (log_q_z - log_p_z), -1)

    # Has an additional KL divergence term for the distribution over alpha
    kl_alpha = gaussian_kl_div(*variational_distribution['alpha'],
                               *prior_distribution['alpha'])

    return -ll + kl_z + kl_alpha


def nll(x, expected_values, covariances):
    # Log likelihood
    ll = loglik_mixture(x.contiguous(),
                        expected_values['z'].contiguous(),
                        covariances)
    return -ll
