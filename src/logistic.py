import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils import loglik, offset, fit_covariances, DataLoader

_eps = 1e-20


def reparam_normal(mu, lv, training=True, size=None):
    if training:
        if size is None:
            eps = Variable(mu.data.new(mu.size()).normal_())
        else:
            eps = Variable(mu.data.new(torch.Size(size)).normal_())
        return eps.mul(torch.exp(lv * 0.5)).add_(mu)
    else:
        return mu


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.n_in = params.n_input
        self.n_out = params.n_latent
        self.n_hidden = params.n_hidden
        self.n_layers = params.n_layers

        # RNN
        self.gru = nn.GRU(self.n_in, self.n_hidden, self.n_layers,
                          dropout=params.dropout, batch_first=True)
        self.l_mu = nn.Linear(self.n_hidden, self.n_out)
        self.l_lv = nn.Linear(self.n_hidden, self.n_out)

        # Weight initialization
        for name, p in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
        nn.init.xavier_uniform_(self.l_mu.weight)
        nn.init.xavier_uniform_(self.l_lv.weight)

    def forward(self, x, h):
        out, h = self.gru(x, h)  # Feed the data into the GRU
        q_mu = self.l_mu(out)
        q_lv = self.l_lv(out)
        return q_mu, q_lv, h

    def init_hidden(self, seq_length):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, seq_length,
                                   self.n_hidden).zero_())


class Prior(nn.Module):
    def __init__(self, params):
        super(Prior, self).__init__()
        self.n_in = params.n_latent
        self.n_out = params.n_latent
        self.n_hidden = params.n_hidden
        self.n_layers = params.n_layers

        # RNN
        self.gru = nn.GRU(self.n_in, self.n_hidden, self.n_layers,
                          dropout=params.dropout, batch_first=True)
        self.l_mu = nn.Linear(self.n_hidden, self.n_out)
        self.p_lv = nn.Parameter(torch.zeros([params.n_latent]).float())

        # Weight initialization
        for name, p in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
        nn.init.xavier_uniform_(self.l_mu.weight)

    def forward(self, z, h):
        # q(z_t|x_{:t})
        out, h = self.gru(z, h)
        p_mu = self.l_mu(out)
        return p_mu, self.p_lv, h

    def init_hidden(self, seq_length):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, seq_length,
                                   self.n_hidden).zero_())


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.temp = params.temp

        # Generative model
        self.prior = Prior(params)
        self.B = nn.Parameter(torch.from_numpy(params.B_init))
        self.W = nn.Parameter(torch.from_numpy(params.W_init))

    def forward(self, q_mu, q_lv, h):
        z = reparam_normal(q_mu, q_lv, self.training)

        # p(z_t|z_{:t-1})
        z_offset = offset(z)
        p_mu, p_lv, h = self.prior(z_offset, h)

        # tilde(z) = logistic(z)
        z_tilde = F.softmax(z / self.temp, dim=-1)

        # Covariances
        c = self.covariance()
        # Add a bit on the diagonal to ensure positive definite
        c += torch.eye(c.shape[-1], dtype=z.dtype, device=z.data.device) * 1e-4
        return c, z, z_tilde, p_mu, p_lv, h

    def covariance(self):
        cov = self.W.matmul(self.B)
        cov = cov.matmul(torch.transpose(cov, -1, -2))
        return cov

    def init_hidden(self, seq_length):
        return self.prior.init_hidden(seq_length)


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


def loss(x, z, c, q_mu, q_lv, p_mu=None, p_lv=None):
    KL = gaussian_kl_div(q_mu, q_lv, p_mu, p_lv)
    LL = loglik(x, z, c)
    elbo = LL - KL  # ELBO / variational free energy
    return -elbo


def train(data, params):
    # Check if a set of covariance matrices has been specified
    covariances = None
    if params.cov_fn is not None:
        cov_path = os.path.join(params.data_dir, params.cov_fn)
        assert os.path.exists(cov_path), '{} does not exist'.format(cov_path)
        # if Matlab path, then load into matlab
        if os.path.splitext(cov_path)[1] == '.mat':
            import scipy.io as sio
            covariances = sio.loadmat(cov_path)['cov']
        else:
            covariances = np.load(cov_path)

    # Get the initial estimates of the covariance matrices
    params.W_init, params.B_init = fit_covariances(
        data, params.n_latent, covariances, params.use_pca_cov_model,
        params.n_pc)

    encoder = Encoder(params)
    decoder = Decoder(params)
    if not params.use_pca_cov_model:
        decoder.W.requires_grad = False

    loader = DataLoader(data, params.seq_length, params.batch_size)

    trainable_params = list(encoder.parameters()) + list(decoder.parameters())
    learning_rate = params.lr
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, trainable_params),
                           lr=learning_rate)

    losses = []
    try:
        for i in range(params.num_epochs):
            if i > 0:
                if i % 10 == 0:
                    learning_rate *= 0.98
                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, trainable_params),
                        lr=learning_rate)
                    print('Learning Rate updated to {}'.format(learning_rate))
                if i % 20 == 0:
                    enc_path = os.path.join(params.checkpoint_dir, 'encoder_{}.pt'.format(i))
                    dec_path = os.path.join(params.checkpoint_dir, 'decoder_{}.pt'.format(i))
                    torch.save(encoder.state_dict(), enc_path)
                    torch.save(decoder.state_dict(), dec_path)

            loader.reset_pointer()

            h = encoder.init_hidden(params.batch_size)
            h_0 = decoder.init_hidden(params.batch_size)

            for b in range(loader.n_id):
                x = loader.next()
                h = Variable(h.data)
                h_0 = Variable(h_0.data)

                q_mu, q_lv, h = encoder(x, h)
                c, z, z_tilde, p_mu, p_lv, h_0 = decoder(q_mu, q_lv, h_0)
                elbo = loss(x, z_tilde, c, q_mu, q_lv, p_mu, p_lv).sum() / params.batch_size

                optimizer.zero_grad()
                elbo.backward()
                nn.utils.clip_grad_norm_(trainable_params, 20)
                optimizer.step()
                losses.append(elbo.data.numpy())

            print('epoch {} Loss {}'.format(i, losses[-1]))

    except KeyboardInterrupt:
        print()

    finally:
        torch.save(encoder.state_dict(), params.checkpoint_dir + '/encoder_final.pt')
        torch.save(decoder.state_dict(), params.checkpoint_dir + '/decoder_final.pt')
        np.save(params.result_dir + '/train_loss.npy', np.array(losses))

    return encoder, decoder, losses


def infer(encoder, decoder, data, params):
    # Load data
    loader = DataLoader(data, params.seq_length, params.batch_size)

    encoder.eval()  # If dropout is being used, turn off for inference
    decoder.eval()

    z_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    z_tilde_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    q_mu_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    q_lv_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    p_mu_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    p_lv_all = np.zeros([loader.n_id, params.batch_size, params.seq_length, params.n_latent])
    loss_all = np.zeros([loader.n_id, params.batch_size, params.seq_length])

    h = encoder.init_hidden(params.batch_size)
    h_0 = decoder.init_hidden(params.batch_size)

    for b in range(loader.n_id):
        x = loader.next()
        h = Variable(h.data);
        h_0 = Variable(h_0.data)

        q_mu, q_lv, h = encoder(x, h)
        c, z, z_tilde, p_mu, p_lv, h_0 = decoder(q_mu, q_lv, h_0)
        nll = -loglik(x, z, c)

        z_all[b] = z.data.cpu().numpy()
        z_tilde_all[b] = z_tilde.data.cpu().numpy()
        q_mu_all[b] = q_mu.data.cpu().numpy()
        q_lv_all[b] = q_lv.data.cpu().numpy()
        p_mu_all[b] = p_mu.data.cpu().numpy()
        p_lv_all[b] = p_mu.data.cpu().numpy()
        loss_all[b] = nll.data.cpu().numpy()

    z_all = np.concatenate(np.concatenate(z_all, 1), 0)
    z_tilde_all = np.concatenate(np.concatenate(z_tilde_all, 1), 0)
    q_mu_all = np.concatenate(np.concatenate(q_mu_all, 1), 0)
    q_lv_all = np.concatenate(np.concatenate(q_lv_all, 1), 0)
    p_mu_all = np.concatenate(np.concatenate(p_mu_all, 1), 0)
    p_lv_all = np.concatenate(np.concatenate(p_lv_all, 1), 0)
    loss_all = np.concatenate(np.concatenate(loss_all, 1), 0)
    covariances = c.data.cpu().numpy()

    np.save(params.result_dir + '/z_all.npy', z_all)
    np.save(params.result_dir + '/z_tilde_all.npy', z_tilde_all)
    np.save(params.result_dir + '/q_mu_all.npy', q_mu_all)
    np.save(params.result_dir + '/q_lv_all.npy', q_lv_all)
    np.save(params.result_dir + '/p_mu_all.npy', p_mu_all)
    np.save(params.result_dir + '/p_lv_all.npy', p_lv_all)
    np.save(params.result_dir + '/loss_all.npy', loss_all)
    np.save(params.result_dir + '/inferred_covariances.npy', covariances)


def main(params):
    np.random.seed(params.rand_seed)
    torch.manual_seed(params.rand_seed)

    data_path = os.path.join(params.data_dir, params.data_fn)
    assert os.path.exists(data_path), '{} does not exist'.format(data_path)
    data = np.load(data_path)

    if not os.path.exists(params.result_dir):
        os.makedirs(params.result_dir)

    params.checkpoint_dir = os.path.join(params.result_dir, 'checkpoint')
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    params.n_input = data.shape[-1]
    encoder, decoder, losses = train(data, params)
    infer(encoder, decoder, data, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    dir_ = os.path.dirname(os.path.realpath(__file__))
    default_data = os.path.join(dir_, 'data')
    default_result = os.path.join(dir_, 'results', str(int(time.time())))

    parser.add_argument('--data-dir',           '-d', default=default_data)
    parser.add_argument('--data-fn',            '-f', default='data.npy')
    parser.add_argument('--cov-fn',             '-c', default=None)
    parser.add_argument('--result-dir',         '-r', default=default_result)
    parser.add_argument('--temp',               '-t', type=float,   default=0.5)
    parser.add_argument('--seq-length',         '-s', type=int,     default=200)
    parser.add_argument('--batch-size',         '-b', type=int,     default=50)
    parser.add_argument('--use-pca-cov-model',  '-p', type=bool,    default=0)
    parser.add_argument('--n-latent',           '-k', type=int,     default=8)
    parser.add_argument('--n-pc',                     type=int,     default=0)
    parser.add_argument('--n-hidden',           '-h', type=int,     default=256)
    parser.add_argument('--n-layers',           '-l', type=int,     default=1)
    parser.add_argument('--dropout',            '-o', type=float,   default=0.)
    parser.add_argument('--num-epochs',         '-e', type=int,     default=100)
    parser.add_argument('--lr',                '-lr', type=float,   default=1e-3)
    parser.add_argument('--rand-seed',         '-rs', type=int,     default=42)
    parser.add_argument('--cuda',               '-x', type=bool,    default=0)

    args = parser.parse_args()
    main(args)
