import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

_eps = 1e-20


def sample_gumbel(shape):
    """
    Sample from the distribution Gumbel(0, 1)
    This can be sampled by taking a rv u ~ Uniform(0, 1) and linearly
    transforming it to get g = -log(-log(u + epsilon)), where epsilon
    ensures numerical stability
    """
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + _eps) + _eps)


def gumbel_softmax(logits, temperature, hard=False):
    """
    Approximately sample from a categorical distribution with parameters alpha
    using the Gumbel softmax.
    Logits x_i = log(/pi_i), gumbel samples g_i (given by sample_gumbel
    function). Tau is a temperature parameter that determines how close to a
    one-hot vector the sample is (closer to one hot as tau > 0). Usually set
    to 0.5

    Compute the sample via the equation:
    y_i = exp((x_i + g_i)/tau)/sum_j(exp((x_i + g_i)/tau))
    """
    gumbel_softmax_sample = logits + sample_gumbel(np.shape(logits)).type(logits.type())
    y = F.softmax(gumbel_softmax_sample / temperature, dim=-1)

    if hard:
        # hard sets the result to a one-hot vector, but retains the gradient
        # of the "soft" variant. Seems to be less effective.
        y_hard = torch.eq(y, y.max(-1, keepdim=True)[0]).type(y.type())
        y = (y_hard - y).detach() + y  # y only element with gradient
    return y


def reparam_normal(mu, lv, training=True, size=None):
    if training:
        if size is None:
            eps = Variable(mu.data.new(mu.size()).normal_())
        else:
            eps = Variable(mu.data.new(torch.Size(size)).normal_())
        return eps.mul(torch.exp(lv * 0.5)).add_(mu)
    else:
        return mu


def log_sum_exp(tensor, keepdim=True):
    """
    Numerically stable implementation for the `LogSumExp` operation. The
    summing is done along the last dimension.
    
    tensor: tensor
    keepdim: Whether to retain the last dimension on summing
    """
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return max_val + (tensor - max_val).exp().sum(dim=-1, keepdim=keepdim).log()


def loglik_mixture(x, z, c):
    """
    Calculate the log likelihood of a mixture of distributions
    """
    x = x.contiguous().view(-1, x.shape[-1]).unsqueeze(0)
    x = x.repeat(c.shape[0], 1, 1)
    from torch.distributions.multivariate_normal import MultivariateNormal
    pdf = MultivariateNormal(
        x.data.new_zeros(x.shape[-1]), c.unsqueeze(1)).log_prob(x).t()
    z_reshape = z.view(-1, c.shape[0]) + _eps
    return log_sum_exp(pdf + z_reshape.log()).reshape(*z.shape[:-1])


def loglik_emissions(x, c):
    """
    Calculate the log likelihood of each emission distribution
    """
    x = x.contiguous().view(-1, x.shape[-1]).unsqueeze(0)
    x = x.repeat(c.shape[0], 1, 1)
    from torch.distributions.multivariate_normal import MultivariateNormal
    return MultivariateNormal(
        x.data.new_zeros(x.shape[-1]), c.unsqueeze(1)).log_prob(x).t()


def gaussian_kl_div(q_mu, q_lv, p_mu=None, p_lv=None):
    if p_mu is None and p_lv is None:
        p_mu = Variable(torch.zeros(q_mu.shape), requires_grad=False)
        p_lv = Variable(torch.zeros(q_lv.shape), requires_grad=False)
    q_v = torch.exp(q_lv)
    p_v = torch.exp(p_lv)
    log_ratio = p_lv - q_lv
    diff = (q_mu - p_mu) ** 2
    # KLdiv for multivariate diagonal gaussians is sum of univariate ones
    return 0.5 * torch.sum(log_ratio + (q_v + diff)/p_v - 1, dim=-1)


def offset(data):
    """
    When we feed the latent factors to the prior, want to offset them 
    forward by one, so that when we calculate the relevant part of the loss 
    (the KL divergence), we are calculating p(z_t|z_{t-1:})

    data: tensor in the shape [batch_size, seq_length, data_dim]
    """
    data_0 = Variable(
        data.data.new(data.size(0), 1, data.size(2)).normal_())
    # Need to softmax to ensure it is a valid distribution
    data_0 = F.softmax(data_0, dim=-1)
    return torch.cat((data_0, data[:, :-1]), dim=1)


def decompose_covariances(c, n_pcs=0, diag_B=False, use_pca_cov_model=False):
    """
    Decompose the covariances into two matrices W and B such that:
    C = WBW'
    Note that if B is a full matrix rather than a diagonal, it is given by
    B = \hat{B}\hat{B}' in order to ensure it is at least positive semidefinite
    
    c: Covariances np.array of shape [latent_dim, data_dim, data_dim]
    n_pcs: number of principal components, int
    diag_B: Whether to use diagonal or full B
    use_pca_cov_model: Whether to set W to identity matrix
    """
    if n_pcs == 0:
        n_pcs = c.shape[-1]

    covmats = np.concatenate(c, 1)

    if diag_B:
        B = np.zeros([c.shape[0], n_pcs])
    else:
        B = np.zeros([c.shape[0], n_pcs, n_pcs])

    if not use_pca_cov_model:
        W = np.eye(c.shape[-1])
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pcs)
        pca.fit(np.transpose(covmats))
        W = np.transpose(pca.components_)

    for kk in range(c.shape[0]):
        from scipy.linalg import sqrtm
        # Bk is obtained by projecting kth cov_X onto the PCs, W:
        Bk = np.matmul(np.linalg.pinv(W), c[kk])
        Bk = sqrtm(np.matmul(Bk, np.transpose(Bk)))

        if diag_B:
            Bk = np.diag(Bk)
            B[kk, :] = np.log(Bk)
        else:
            B[kk, :, :] = np.linalg.cholesky(Bk)
    return W.astype(np.float32), B.astype(np.float32)


def fit_covariances(data, n_mixtures, C=None, use_pca_cov_model=False, n_pcs=0):
    """
    Fit the data using a GMM and decompose the covariance matrices
    
    data: np.array of size [T, data_dim] 
    n_mixtures: size of latent dimension
    use_pca_cov_model: Whether to apply dimensionality reduction on cov matrices
    n_pcs: number of principal components to take in dim. reduction
    """
    if C is None:
        from sklearn.mixture import GaussianMixture
        GMM = GaussianMixture(covariance_type='full', n_components=n_mixtures)
        if len(data.shape) > 2:
            data = data[0]  # if multiple subjects, just take the first subject.
        GMM.fit(data)
        C = GMM.covariances_

        # Add noise to estimates so we don't end up with duplicated states
        noise = np.random.normal(0, 0.1, size=[n_mixtures, data.shape[-1]])
        noise = np.matmul(noise.reshape(n_mixtures, data.shape[-1], 1),
                          noise.reshape(n_mixtures, 1, data.shape[-1]))
        noise += np.random.random(size=[n_mixtures, 1, data.shape[1]])\
                 + np.eye(data.shape[-1]) * 0.3

        C += noise

    if n_pcs > 0:
        W, B = decompose_covariances(C, data.shape[-1], False,
                                     use_pca_cov_model)
    else:
        W, B = decompose_covariances(C, n_pcs, False,
                                     use_pca_cov_model)
    return W, B


class DataLoader(object):
    """
    Class allowing for easy iteration through the data
    """
    def __init__(self, data, seq_length, batch_dim=20, overlap=0):
        """
        Feed in the data and length of each batch
        data: [n x t x d] dimensional data
        batch_length: integer
        """
        if len(data.shape) == 2:
            # If single subject data is fed in, still needs first dimension
            data = np.expand_dims(data, 0)
        n, _, d = data.shape

        # How many times do we need to split each subject to achieve at
        # least batch_size number of mini-batches?
        rep_num = np.ceil(batch_dim / n)
        batch_dim = rep_num * n

        # Find the smallest non-padded length
        min_t = np.min([np.any(datum != 0., axis=-1).sum() for datum in data])
        # Rather than bothering with padding, just trim the excess off
        trimmed_t = int(min_t - (min_t - (overlap * rep_num)) % (seq_length * rep_num))
        new_data = data[:, :trimmed_t]

        t = new_data.shape[1]
        n_batches = int((t - (overlap * rep_num)) / (seq_length * rep_num))
        batched_data = np.concatenate(np.split(new_data, rep_num, 1))

        # Reorder the subject indices so instead of being arranged like
        # [1 .. n .. 1 .. n] its [1 .. 1 2 .. 2 .. n .. n]
        subj_ind = (
            np.expand_dims(np.arange(n), -1) + np.arange(rep_num) * n
        ).flatten().astype(np.int32)
        reordered_data = batched_data[subj_ind]

        self.t = t
        self.overlap = overlap
        self.seq_length = seq_length
        self.batch_dim = int(batch_dim)
        self.n = n  # Number of subjects
        self.d = d  # Number of features
        self.n_batches = n_batches
        self.subject_indices = subj_ind
        self.data = reordered_data

        self.idx = 0

    def reset_pointer(self):
        self.idx = 0

    def next(self):
        """
        Returns the next batch of data (in order)
        return: [1 x batch_length x d] dim tensor
        """
        if self.idx == self.n_batches:
            self.reset_pointer()

        start = self.idx * self.seq_length
        end = (self.idx + 1) * self.seq_length + self.overlap
        data = self.data[:, start:end]

        self.idx += 1
        return Variable(torch.from_numpy(data.astype(np.float32)))
