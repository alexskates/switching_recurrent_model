# import os
# import time
# import datetime
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from scipy.io import savemat
# from src.utils import gumbel_softmax, reparam_normal, loglik_mixture, offset, \
#     fit_covariances, DataLoader, gaussian_kl_div
# from src.modules import TCNInference, RNNInference, TCNPrior, RNNPrior, \
#     MarkovPrior, UninformativePrior
#
# _eps = 1e-20
#
#
# class Encoder(nn.Module):
#     def __init__(self, params):
#         super(Encoder, self).__init__()
#
#         if params.inference.lower() == 'tcn':
#             self.inference = TCNInference(params)
#         else:
#             self.inference = RNNInference(params)
#
#         if params.bidirectional:
#             self.l_mu = nn.Linear(params.n_hidden * 2, params.n_latent)
#             self.l_lv = nn.Linear(params.n_hidden * 2, params.n_latent)
#         else:
#             self.l_mu = nn.Linear(params.n_hidden, params.n_latent)
#             self.l_lv = nn.Linear(params.n_hidden, params.n_latent)
#         nn.init.xavier_uniform_(self.l_mu.weight)
#         nn.init.xavier_uniform_(self.l_lv.weight)
#
#         if params.cuda:
#             self.cuda()
#
#     def forward(self, x, h=None):
#         if type(self.inference) == RNNInference:
#             q_alpha_raw, _, h = self.inference(x, h)
#         else:
#             q_alpha_raw, _ = self.inference(x)
#
#         # Introduce extra stochastic layer compared to gumbel
#         # Sample from normal distribution
#         q_alpha_mu = self.l_mu(q_alpha_raw)
#         q_alpha_lv = self.l_lv(q_alpha_raw)
#         return q_alpha_mu, q_alpha_lv, h
#
#     def init_hidden(self, seq_length):
#         if type(self.inference) == RNNInference:
#             return self.inference.init_hidden(seq_length)
#         else:
#             return None
#
#
# class Decoder(nn.Module):
#     def __init__(self, params):
#         super(Decoder, self).__init__()
#
#         # Generative model
#         if params.prior.lower() == 'rnn':
#             self.prior = RNNPrior(params)
#         elif params.prior.lower() == 'tcn':
#             self.prior = TCNPrior(params)
#         elif params.prior.lower() == 'markov':
#             self.prior = MarkovPrior(params)
#         else:
#             self.prior = UninformativePrior(params)
#
#         self.B = nn.Parameter(torch.from_numpy(params.B_init))
#         self.W = nn.Parameter(torch.from_numpy(params.W_init))
#
#         self.l_mu = nn.Linear(params.n_hidden, params.n_latent)
#         self.l_lv = nn.Linear(params.n_hidden, params.n_latent)
#         nn.init.xavier_uniform_(self.l_mu.weight)
#         nn.init.xavier_uniform_(self.l_lv.weight)
#
#         if params.cuda:
#             self.cuda()
#
#     def forward(self, tau, q_alpha_mu, q_alpha_lv, h=None):
#         # Sample from the inferred alpha parameters
#         alpha = reparam_normal(q_alpha_mu, q_alpha_lv, self.training)
#         # Softmax alpha to get alpha tilde
#         q_alpha_tilde = F.softmax(alpha, dim=-1)
#         # Unable to sample from q_alpha_tilde so use GS to sample from
#         # approximations of it
#         alpha_tilde = gumbel_softmax(alpha, tau, hard=False)
#
#         # p(z_t|z_{:t-1})
#         alpha_offset = offset(alpha)
#         # If we need to pass in a hidden state to the generative model,
#         # the class will have an attribute specifying how big it it
#         if type(self.prior) == RNNPrior:
#             p_alpha_raw, _, _, _, h = self.prior(alpha_offset, h)
#         else:
#             p_alpha_raw, _, _, _ = self.prior(alpha_offset)
#
#         p_alpha_mu = self.l_mu(p_alpha_raw)
#         p_alpha_lv = self.l_lv(p_alpha_raw)
#         alpha_0 = reparam_normal(p_alpha_mu, p_alpha_lv, self.training)
#         p_alpha_tilde = F.softmax(alpha_0, dim=-1)
#
#         # Covariances
#         c = self.covariance()
#         # Ensure is positive definite
#         c += torch.eye(c.shape[-1], dtype=alpha.dtype, device=alpha.data.device) * 1e-4
#         return c, alpha, alpha_0, p_alpha_mu, p_alpha_lv, alpha_tilde, \
#                q_alpha_tilde, p_alpha_tilde, h
#
#     def covariance(self):
#         cov = self.W.matmul(self.B)
#         cov = cov.matmul(torch.transpose(cov, -1, -2))
#         return cov
#
#     def init_hidden(self, seq_length):
#         if type(self.prior) == RNNPrior:
#             return self.prior.init_hidden(seq_length)
#         else:
#             return None
#
#
# def loss(x, alpha_tilde, c, q_alpha_mu, q_alpha_lv, p_alpha_mu, p_alpha_lv,
#          q_alpha_tilde, p_alpha_tilde):
#     kl_alpha = gaussian_kl_div(q_alpha_mu, q_alpha_lv, p_alpha_mu, p_alpha_lv)
#
#     log_q = torch.log(q_alpha_tilde + _eps)
#     log_p = torch.log(p_alpha_tilde + _eps)
#     kl_alpha_tilde = torch.sum(q_alpha_tilde * (log_q - log_p), -1)
#
#     loglik = loglik_mixture(x, alpha_tilde, c)
#     elbo = loglik - (kl_alpha + kl_alpha_tilde)
#     return -elbo
#
#
# def init(data, params):
#     # Check if a set of covariance matrices has been specified
#     covariances = None
#     if params.cov_fn is not None:
#         cov_path = os.path.join(params.data_dir, params.cov_fn)
#         assert os.path.exists(cov_path), '{} does not exist'.format(
#             cov_path)
#         # if Matlab path, then load into matlab
#         if os.path.splitext(cov_path)[1] == '.mat':
#             import scipy.io as sio
#             covariances = sio.loadmat(cov_path)['cov']
#         else:
#             covariances = np.load(cov_path)
#
#     # Get the initial estimates of the covariance matrices
#     params.W_init, params.B_init = fit_covariances(
#         data, params.n_latent, covariances, params.use_pca_cov_model,
#         params.n_pc)
#
#     encoder = Encoder(params)
#     decoder = Decoder(params)
#     if not params.use_pca_cov_model:
#         decoder.W.requires_grad = False
#
#     # Look to see if a checkpoint has already been specified
#     if (params.encoder_fn is not None) and (params.decoder_fn is not None):
#         encoder_path = os.path.join(params.checkpoint_dir, params.encoder_fn)
#         decoder_path = os.path.join(params.checkpoint_dir, params.decoder_fn)
#         assert os.path.exists(encoder_path), '{} does not exist'.format(encoder_path)
#         assert os.path.exists(decoder_path), '{} does not exist'.format(decoder_path)
#         # Load the checkpoint
#         encoder.load_state_dict(torch.load(encoder_path, map_location={'cuda:0': 'cpu'}))
#         decoder.load_state_dict(torch.load(decoder_path, map_location={'cuda:0': 'cpu'}))
#
#     return encoder, decoder
#
#
# def train(encoder, decoder, data, params):
#     loader = DataLoader(data, params.seq_length, params.batch_size, overlap=params.overlap)
#
#     learning_rate = params.lr
#     trainable_params = list(encoder.parameters()) + list(decoder.parameters())
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, trainable_params),
#                            lr=learning_rate)
#
#     start = datetime.datetime.now()
#     print(
#         '{:%d, %b %Y %H:%M}\n'
#         'Data file: {}\n'
#         'Data dim: {}\n'
#         '------------\n\n'.format(
#             start, os.path.join(params.data_dir, params.data_fn),
#             data.shape))
#
#     temp = torch.Tensor([params.temp])
#     if params.cuda:
#         temp = temp.cuda()
#
#     losses = []
#     try:
#         for i in range(params.num_epochs):
#             if i > 0:
#                 if i % params.checkpoint_num == 0:
#                     enc_path = os.path.join(
#                         params.checkpoint_dir, 'encoder_{}.pt'.format(i))
#                     dec_path = os.path.join(
#                         params.checkpoint_dir, 'decoder_{}.pt'.format(i))
#                     torch.save(encoder.state_dict(), enc_path)
#                     torch.save(decoder.state_dict(), dec_path)
#
#             loader.reset_pointer()
#
#             h = encoder.init_hidden(loader.batch_dim)
#             h_0 = decoder.init_hidden(loader.batch_dim)
#
#             epoch_loss = []
#             for b in range(loader.n_batches):
#                 x = loader.next()
#                 if h is not None: h = Variable(h.data)
#                 if h_0 is not None: h_0 = Variable(h_0.data)
#
#                 if params.cuda:
#                     x = x.cuda()
#                     if h is not None: h = h.cuda()
#                     if h_0 is not None: h_0 = h_0.cuda()
#
#                 q_alpha_mu, q_alpha_lv, h = encoder(x, h)
#                 c, alpha, alpha_0, p_alpha_mu, p_alpha_lv, alpha_tilde, \
#                 q_alpha_tilde, p_alpha_tilde, h_0 = decoder(temp, q_alpha_mu, q_alpha_lv, h_0)
#
#                 elbo = loss(x[:, params.overlap:].contiguous(),
#                             alpha_tilde[:, params.overlap:].contiguous(),
#                             c,
#                             q_alpha_mu[:, params.overlap:].contiguous(),
#                             q_alpha_lv[:, params.overlap:].contiguous(),
#                             p_alpha_mu[:, params.overlap:].contiguous(),
#                             p_alpha_lv[:, params.overlap:].contiguous(),
#                             q_alpha_tilde[:, params.overlap:].contiguous(),
#                             p_alpha_tilde[:, params.overlap:].contiguous()).sum()
#                 elbo = elbo / params.batch_size
#
#                 epoch_loss.append(elbo.data.cpu().numpy())
#                 optimizer.zero_grad()
#                 elbo.backward()
#                 nn.utils.clip_grad_norm_(trainable_params, 10)
#                 optimizer.step()
#
#             losses.append(np.mean(epoch_loss))
#             print('epoch {} Loss {}'.format(i, losses[-1]))
#
#     except KeyboardInterrupt:
#         print('Keyboard Interrupt')
#     except Exception as e:
#         print(e)
#     finally:
#         torch.save(encoder.state_dict(), params.checkpoint_dir + '/encoder_final.pt')
#         torch.save(decoder.state_dict(), params.checkpoint_dir + '/decoder_final.pt')
#         np.save(params.result_dir + '/train_loss.npy', np.array(losses))
#
#     print('Training complete, final loss: {}, time taken: {}\n'.format(
#             losses[-1], str(datetime.datetime.now() - start)))
#
#
# def infer(encoder, decoder, data, params):
#     # How many subjects?
#     if len(data.shape) > 2:
#         n_subjects = data.shape[0]
#     else:
#         n_subjects = 1
#
#     # If dropout is being used, turn off
#     encoder.eval()
#     decoder.eval()
#
#     loader = DataLoader(data, params.seq_length, n_subjects, overlap=params.overlap)
#
#     q_alpha_mu_subject = np.zeros([n_subjects, loader.t, params.n_latent])
#     q_alpha_lv_subject = np.zeros([n_subjects, loader.t, params.n_latent])
#     q_alpha_tilde_subject = np.zeros([n_subjects, loader.t, params.n_latent])
#     losses = np.zeros([n_subjects, loader.t - params.overlap])
#
#     h = encoder.init_hidden(loader.batch_dim)
#     h_0 = decoder.init_hidden(loader.batch_dim)
#
#     temp = torch.Tensor([params.temp])
#     if params.cuda:
#         temp = temp.cuda()
#
#     with torch.no_grad():
#         for b in range(loader.n_batches):
#             x = loader.next()
#             if h is not None: h = Variable(h.data)
#             if h_0 is not None: h_0 = Variable(h_0.data)
#
#             if params.cuda:
#                 x = x.cuda()
#                 if h is not None: h = h.cuda()
#                 if h_0 is not None: h_0 = h_0.cuda()
#
#             q_alpha_mu, q_alpha_lv, h = encoder(x, h)
#             c, alpha, alpha_0, p_alpha_mu, p_alpha_lv, alpha_tilde, q_alpha_tilde, \
#             p_alpha_tilde, h_0 = decoder(temp, q_alpha_mu, q_alpha_lv, h_0)
#
#             # Have to reshape the output
#             if b == 0:
#                 elbo = loss(x, alpha_tilde, c, q_alpha_mu, q_alpha_lv, p_alpha_mu,
#                             p_alpha_lv, q_alpha_tilde, p_alpha_tilde).sum()
#
#                 idx = params.overlap + params.seq_length
#                 q_alpha_mu_subject[:, :idx] = q_alpha_mu.data.cpu().numpy()
#                 q_alpha_lv_subject[:, :idx] = q_alpha_lv.data.cpu().numpy()
#                 q_alpha_tilde_subject[:, :idx] = q_alpha_tilde.data.cpu().numpy()
#                 losses[:, :idx] = elbo.data.cpu().numpy()
#             else:
#                 elbo = loss(x[:, params.overlap:].contiguous(),
#                             alpha_tilde[:, params.overlap:].contiguous(),
#                             c,
#                             q_alpha_mu[:, params.overlap:].contiguous(),
#                             q_alpha_lv[:, params.overlap:].contiguous(),
#                             p_alpha_mu[:, params.overlap:].contiguous(),
#                             p_alpha_lv[:, params.overlap:].contiguous(),
#                             q_alpha_tilde[:, params.overlap:].contiguous(),
#                             p_alpha_tilde[:, params.overlap:].contiguous()).sum()
#
#                 idx_start = b * params.seq_length
#                 idx_end = (b + 1) * params.seq_length
#                 q_alpha_mu_subject[:, idx_start:idx_end] = q_alpha_mu[:, params.overlap:].data.cpu().numpy()
#                 q_alpha_lv_subject[:, idx_start:idx_end] = q_alpha_lv[:, params.overlap:].data.cpu().numpy()
#                 q_alpha_tilde_subject[:, idx_start:idx_end] = q_alpha_tilde[:, params.overlap:].data.cpu().numpy()
#                 losses[:, idx_start:idx_end] = elbo.data.cpu().numpy()
#
#     covariances = decoder.covariance().data.cpu().numpy()
#
#     np.save(params.result_dir + '/q_z_subject.npy', q_alpha_tilde_subject)
#     np.save(params.result_dir + '/q_alpha_mu_subject.npy', q_alpha_mu_subject)
#     np.save(params.result_dir + '/q_alpha_lv_subject.npy', q_alpha_lv_subject)
#     np.save(params.result_dir + '/vfe_subject.npy', losses)
#     np.save(params.result_dir + '/q_z_all.npy', np.concatenate(q_alpha_tilde_subject, 0))
#     np.save(params.result_dir + '/q_alpha_mu_all.npy', np.concatenate(q_alpha_mu_subject, 0))
#     np.save(params.result_dir + '/q_alpha_lv_all.npy', np.concatenate(q_alpha_lv_subject, 0))
#     np.save(params.result_dir + '/vfe_all.npy', np.concatenate(losses, 0))
#     np.save(params.result_dir + '/inferred_covariances.npy', covariances)
#     savemat(params.result_dir + '/inferred_covariances.mat', {'C': covariances})
#     if params.prior.lower() == 'markov':
#         np.save(params.result_dir + '/trans_prob.npy',
#                 decoder.prior.trans_prob().data.numpy())
#
#
# def main(params):
#     np.random.seed(params.rand_seed)
#     torch.manual_seed(params.rand_seed)
#
#     assert params.prior.lower() in ['rnn', 'markov', 'uninformative', 'tcn'], \
#         'Prior {} is not implemented'.format(params.prior)
#     assert params.inference.lower() in ['rnn', 'tcn'], \
#         'Inference {} is not implemented'.format(params.inference)
#     if params.prior.lower() == 'tcn' or params.inference.lower() == 'tcn':
#         # Want a "burn in" period for each batch that allows every inferred
#         # timepoint to benefit from a the receptive field
#         discard_data = 2 * params.n_kernel * 2 ** (params.n_layers - 1)
#         params.overlap = discard_data
#     else:
#         params.overlap = 0
#
#     data_path = os.path.join(params.data_dir, params.data_fn)
#     assert os.path.exists(data_path), '{} does not exist'.format(data_path)
#     data = np.load(data_path)
#
#     if not os.path.exists(params.result_dir):
#         os.makedirs(params.result_dir)
#
#     params.checkpoint_dir = os.path.join(params.result_dir, 'checkpoint')
#     if not os.path.exists(params.checkpoint_dir):
#         os.makedirs(params.checkpoint_dir)
#
#     # Save model parameters
#     import json
#     with open(os.path.join(params.result_dir, 'params.json'), 'w') as outfile:
#         json.dump(params.__dict__, outfile)
#
#     params.n_input = data.shape[-1]
#     encoder, decoder = init(data, params)
#     if params.train:
#         train(encoder, decoder, data, params)
#
#     if params.infer:
#         infer(encoder, decoder, data, params)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(add_help=False)
#
#     dir_ = os.path.dirname(os.path.realpath(__file__))
#     default_data = os.path.join(dir_, 'data')
#     default_result = os.path.join(dir_, 'results', str(int(time.time())))
#
#     parser.add_argument('--data-dir',           '-d', default=default_data)
#     parser.add_argument('--data-fn',            '-f', default='data.npy')
#     parser.add_argument('--encoder-fn',       '-enc', default=None)
#     parser.add_argument('--decoder-fn',       '-dec', default=None)
#     parser.add_argument('--cov-fn',             '-c', default=None)
#     parser.add_argument('--result-dir',         '-r', default=default_result)
#     parser.add_argument('--temp',               '-t', type=float,   default=0.5)
#     parser.add_argument('--seq-length',         '-s', type=int,     default=200)
#     parser.add_argument('--batch-size',         '-b', type=int,     default=50)
#     parser.add_argument('--use-pca-cov-model',  '-p', action='store_true')
#     parser.add_argument('--n-latent',           '-k', type=int,     default=8)
#     parser.add_argument('--n-pc',              '-pc', type=int,     default=0)
#     parser.add_argument('--n-hidden',           '-h', type=int,     default=256)
#     parser.add_argument('--n-layers',           '-l', type=int,     default=1)
#     parser.add_argument('--n-kernel',           '-K', type=int,     default=5)
#     parser.add_argument('--dropout',            '-o', type=float,   default=0.)
#     parser.add_argument('--num-epochs',         '-e', type=int,     default=100)
#     parser.add_argument('--checkpoint-num',    '-ck', type=int,     default=20)
#     parser.add_argument('--lr',                '-lr', type=float,   default=1e-3)
#     parser.add_argument('--rand-seed',         '-rs', type=int,     default=42)
#     parser.add_argument('--mask',               '-m', action='store_true')
#     parser.add_argument('--cuda',               '-x', action='store_true')
#     parser.add_argument('--bidirectional',     '-bi', action='store_true')
#     parser.add_argument('--prior',             '-pr', default='rnn')
#     parser.add_argument('--inference',        '-inf', default='rnn')
#     parser.add_argument('--train',             '-tr', action='store_true')
#     parser.add_argument('--infer',              '-i', action='store_true')
#
#     args = parser.parse_args()
#     main(args)
