import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import reverse_sequence

_eps = 1e-20


# # Modules for performing amortized inference --
# # Need to be able to sample from them
# class TCNDistribution(nn.Module):
#     def __init__(self, params):
#         super(TCNDistribution, self).__init__()
#         self.n_in = params.n_input
#         self.n_out = params.n_latent
#         self.n_kernel = params.n_kernel
#         self.n_layers = params.n_layers
#         self.n_channels = [params.n_hidden] * self.n_layers
#
#         self.tcn = TemporalConvNet(self.n_in, self.n_channels, self.n_kernel,
#                                    dropout=params.dropout)
#         self.l_logits = nn.Linear(self.n_channels[-1], self.n_out)
#
#         # Weight initialization
#         for name, p in self.tcn.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_normal_(p)
#         nn.init.xavier_uniform_(self.l_logits.weight)
#
#         if params.cuda:
#             self.cuda()
#
#     def forward(self, inputs):
#         x, _ = inputs
#         # X needs to have dimension [batch, channels, length]
#         out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
#         return out, self.l_logits(out), None


class RNNDistribution(nn.Module):
    def __init__(self, params):
        super(RNNDistribution, self).__init__()
        self.bidirectional = params.bidirectional
        self.backwards = params.backwards_inf
        self.n_in = params.n_input
        self.n_out = params.n_latent
        self.n_hidden = params.n_hidden
        self.n_layers = params.n_layers

        # I'm using a GRU here, but could just as easily be an LSTM etc.
        self.gru = nn.GRU(self.n_in, self.n_hidden, self.n_layers,
                          dropout=params.dropout, batch_first=True,
                          bidirectional=self.bidirectional)

        self.num_directions = 2 if params.bidirectional else 1
        self.l_logits = nn.Linear(self.n_hidden * self.num_directions, self.n_out)

        # Weight initialization
        for name, p in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
        nn.init.xavier_uniform_(self.l_logits.weight)

        if params.cuda:
            self.cuda()

    def forward(self, inputs):
        x, h = inputs
        if (not self.bidirectional) and self.backwards:
            # If we want to train the network to output q(z_t | x_{<=t})
            x_lengths = [x.shape[1]] * x.shape[0]
            x_reversed = reverse_sequence(x, x_lengths, True)
            bwards_output_reversed, h = self.gru(x_reversed, h)
            out = reverse_sequence(bwards_output_reversed, x_lengths, True)
        else:
            out, h = self.gru(x, h)
        return out, self.l_logits(out), h

    def init_hidden(self, batch_size):
        # Initialise all the hidden states in the inference network
        weight = next(self.parameters()).data
        if self.bidirectional:
            return Variable(weight.new(self.n_layers*2, batch_size,
                                       self.n_hidden).zero_())
        else:
            return Variable(weight.new(self.n_layers, batch_size,
                                       self.n_hidden).zero_())


# Modules that don't require sampling from
class RNNSoftmax(nn.Module):
    def __init__(self, params):
        super(RNNSoftmax, self).__init__()
        self.n_in = params.n_latent
        self.n_out = params.n_latent
        self.n_hidden = params.n_hidden
        self.n_layers = params.n_layers

        # RNN
        self.gru = nn.GRU(self.n_in, self.n_hidden, self.n_layers,
                          dropout=params.dropout, batch_first=True)
        self.l_logits = nn.Linear(self.n_hidden, self.n_out)

        # Weight initialization
        for name, p in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(p)
        nn.init.xavier_uniform_(self.l_logits.weight)

        if params.cuda:
            self.cuda()

    def forward(self, inputs):
        z, h = inputs
        # q(z_t|x_{:t})
        out, h = self.gru(z, h)
        logits_z = self.l_logits(out)
        return logits_z, F.softmax(logits_z, dim=-1), h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, batch_size,
                                   self.n_hidden).zero_())

#
# class TCNSoftmax(nn.Module):
#     def __init__(self, params):
#         super(TCNSoftmax, self).__init__()
#         self.n_in = params.n_latent
#         self.n_out = params.n_latent
#         self.n_kernel = params.n_kernel
#         self.n_layers = params.n_layers
#         self.n_channels = [params.n_hidden] * self.n_layers
#
#         self.tcn = TemporalConvNet(self.n_in, self.n_channels, self.n_kernel,
#                                    dropout=params.dropout)
#         self.l_logits = nn.Linear(self.n_channels[-1], self.n_out)
#
#         # Weight initialization
#         for name, p in self.tcn.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_normal_(p)
#         nn.init.xavier_uniform_(self.l_logits.weight)
#
#         if params.cuda:
#             self.cuda()
#
#     def forward(self, inputs):
#         z, _ = inputs
#         # q(z_t|x_{:t})
#         # z needs to have dimension [batch, channels, length]
#         out = self.tcn(z.transpose(1, 2)).transpose(1, 2)
#         logits_z = self.l_logits(out)
#         return logits_z, F.softmax(logits_z, dim=-1), None
#
#     def init_hidden(self, batch_size):
#         return None


class MarkovSoftmax(nn.Module):
    def __init__(self, params):
        super(MarkovSoftmax, self).__init__()
        self.n_latent = params.n_latent

        # Transition Probability Matrix
        # Initialise it with uniform probabilities
        # (softmax it below)
        self.t = nn.Parameter(torch.ones(params.n_latent, params.n_latent) +
                              torch.eye(params.n_latent) * params.prior_diag)
        self.t.requires_grad = True
        self.requires_softmax = True

        if params.cuda:
            self.cuda()

    def forward(self, inputs):
        z, _ = inputs
        # q(z_t|x_{:t})
        # z should be pretty much one-hot, meaning matmul acts to only get
        # the probabilities at index argmax(z[t])
        p_z = z.matmul(self.trans_prob())
        return p_z.log(), p_z, None

    def trans_prob(self):
        # Ensure the transition probability matrix sums to 1
        if self.requires_softmax:
            return F.softmax(self.t, dim=1)
        else:
            return self.t

    def init_hidden(self, batch_size):
        return None


class Uninform(nn.Module):
    def __init__(self, params):
        super(Uninform, self).__init__()
        self.n_latent = params.n_latent

        # Transition Probability Matrix
        self.Uniform = Variable(torch.ones(params.n_latent) / params.n_latent)
        self.Uniform.requires_grad = False

        if params.cuda:
            self.Uniform = self.Uniform.cuda()

    def forward(self, inputs):
        z, _ = inputs
        p_z = self.Uniform
        return _, p_z, None

    def init_hidden(self, batch_size):
        return None
