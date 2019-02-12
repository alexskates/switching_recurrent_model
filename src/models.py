import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import gumbel_softmax, loglik_mixture, offset, reparam_normal, gaussian_kl_div
from src.modules import TCNDistribution, RNNDistribution, RNNSoftmax, TCNSoftmax,\
    MarkovSoftmax, Uninform

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


class HierarchicalDecoder(nn.Module):
    def __init__(self, params):
        super(GumbelDecoder, self).__init__()

        # Generative model
        if params.prior.lower() == 'rnn':
            params.bidirectional = False
            params.n_layers = 1
            self.prior = [
                RNNDistribution(params) for _ in range(params.n_hierarchies)
            ]

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
