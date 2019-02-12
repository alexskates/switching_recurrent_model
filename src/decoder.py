import heapq
import torch
import numpy as np
from torch.autograd import Variable


class Sequence(object):
    """
    Represents a potential sequence.
    """

    def __init__(self, sequence, state, logprob):
        """
        Initialises the sequences.
        """
        self.sequence = sequence
        self.state = state
        self.logprob = logprob

    def __repr__(self):
        return '<Sequence: {}, logprob: {:.3f}>'.format(self.sequence, self.logprob)

    def __lt__(self, other_sequence):
        """
        Compares Sequence with another sequence. We can then find the top n
        most likely sequences in a list of Sequence objects using the following:

        import heapq
        top_n = heapq.nlargest(n, queue)
        """
        assert isinstance(other_sequence, Sequence)
        return self.logprob < other_sequence.logprob


class PriorityQueue:
    """
    Priority queue to place sequences in. Maintains a capacity of at most n
    items.
    """

    def __init__(self, n):
        self._n = n
        self._items = []

    def size(self):
        assert self._items is not None
        return len(self._items)

    def push(self, x):
        """
        Add a new sequence to the queue
        """
        assert self._items is not None
        if len(self._items) < self._n:
            heapq.heappush(self._items, x)
        else:
            # PushPop will add a new sequence to _items, removing the lowest
            heapq.heappushpop(self._items, x)

    def extract(self):
        """
        Get all items in the queue. Will subsequently remove them
        """
        assert self._items is not None
        items = self._items
        self._items = None
        return items

    def top(self):
        """
        Get the most likely sequence.
        """
        assert self._items is not None
        return heapq.nlargest(1, self._items)[0]

    def reset(self):
        self._items = []


class SequenceDecoder(object):
    """
    Class to infer the most likely sequence of states using an adapted form
    of the algorithm found in "Phone Sequence Modeling with Recurrent Neural
    Networks"
    Found at: http://www-etud.iro.umontreal.ca/~boulanni/ICASSP2014.pdf

    Note that it will not necessarily find the globally optimal solution as
    that would require an exhaustive search of complexity K^T
    """

    def __init__(self, q_z, decoder, T, params):
        self.decoder = decoder
        self.decoder_network = self.decoder.prior
        self.q_z = torch.from_numpy(q_z.astype(np.float32))

        self.T = T

        self.n_states = params.n_latent
        self.beam_width = params.n_latent
        self.cuda = params.cuda
        self.max_seq_length = params.seq_length

    def dynamic_beam_search(self):
        """
        Run the beam search
        """
        with torch.no_grad():
            sequence_list = PriorityQueue(self.beam_width)

            # Get the first candidates. No initial state probabilities implemented,
            # so simply going off of the inferred state probabilities
            for k in range(self.n_states):
                seq = Sequence(
                    sequence=torch.zeros(
                        1, self.n_states).scatter_(1, torch.tensor([[k]]), 1),
                    state=self.decoder_network.init_hidden(1),
                    logprob=self.q_z[0, k].log())
                sequence_list.push(seq)

            # If we are using a HMM or something, don't need states:
            state_feed = [seq.state for seq in sequence_list._items]
            # requires_hidden = state_feed.count(None) != len(state_feed)
            requires_hidden = True

            # Equation (11), factor over time
            for t in range(2, self.T):
                sequences = sequence_list.extract()
                sequence_list.reset()

                # Equation (12) requires us to calculate p(z_t|x)p(z_t|z_{1:t-1})
                # Calculating p(z_t|x) just requires us to run a forward pass
                # through the inference network. This should have already been
                # saved in the "infer" function as "q_z_all.npy"
                log_prob_inference = self.q_z[t].log()

                # Then have to calculate p(z_t|z_{1:t-1})
                # We have K different candidate sequences for z_{1:t-1}.
                # Can feed them all into the prior network in one go with batching
                input_feed = [seq.sequence[-self.max_seq_length:] for seq in sequences]
                state_feed = [seq.state for seq in sequences]
                logprob_feed = [seq.logprob for seq in sequences]

                # Inputs have the batch in 0th dim, states have batch in the 1st
                input_feed = Variable(torch.stack(input_feed))
                state_feed = Variable(torch.cat(state_feed, dim=1))
                logprob_feed = torch.stack(logprob_feed).unsqueeze(-1)

                # Prior step
                _, _, _, log_prob_transition, h = self.decoder_network(input_feed, state_feed)

                # Equation (18), used to figure out which of the K sequences gives
                # the best log likelihood for transitioning into that state
                loglik_sum = logprob_feed + log_prob_transition[:, -1] + log_prob_inference

                for j in range(self.n_states):
                    # Get the index of the sequence
                    l_j_t, seq_idx = torch.max(loglik_sum[:, j], 0)
                    current_state = torch.zeros(
                        1, self.n_states).scatter_(1, torch.tensor([[j]]), 1)
                    # if requires_hidden:
                    seq = Sequence(
                        sequence=torch.cat([sequences[seq_idx].sequence, current_state], dim=0),
                        state=h[:, seq_idx].unsqueeze(1),
                        logprob=l_j_t)
                    # else:
                    #     seq = Sequence(
                    #         sequence=torch.cat([sequences[seq_idx].sequence, current_state], dim=0),
                    #         state=None,
                    #         logprob=l_j_t)
                    sequence_list.push(seq)

                sequences = None

        return sequence_list

# def main():
#     path = '/Users/askates/Documents/Projects/experiments/Simulations' \
#            '/markov_test/results/rnn'
#
#     # Load params
#     import json
#     with open(os.path.join(path, 'params.json')) as f:
#         params = json.load(f)
#     params['train'] = False
#     params['infer'] = False
#
#     # Convert params dictionary to an object
#     class ParamDict2Obj(object):
#         def __init__(self, dictionary):
#             for key in dictionary:
#                 setattr(self, key, dictionary[key])
#
#     params = ParamDict2Obj(params)
#
#     # Load data
#     data = np.load(os.path.join(params.data_dir, params.data_fn))
#     params.n_input = data.shape[-1]
#
#     # Load encoder/decoder
#     encoder, decoder = init(data, params)
#
#     # Load q(z|x)
#     q_z_x = np.load(os.path.join(params.result_dir, 'q_z_all.npy'))
#
#     # trans_prob = np.load(os.path.join(params.data_dir, 'trans_prob.npy'))
#     #
#     # def t():
#     #     tr = torch.nn.Parameter(torch.from_numpy(trans_prob.astype(np.float32)))
#     #     return tr
#
#     # decoder.prior.trans_prob = t
#
#     state_seq = np.load(os.path.join(params.data_dir, 'state_seq.npy'))
#
#     tmp_decoder = SequenceDecoder(q_z_x, decoder, data.shape[0], params)
#     sequence_list = tmp_decoder.dynamic_beam_search()
#
#
# if __name__ == '__main__':
#     main()
#
#



