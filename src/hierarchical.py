import os
import time
import argparse
import numpy as np
import torch
import json
from utils import fit_covariances
from models import (
    HierarchicalEncoder,
    FactoredHierarchicalEncoder,
    HierarchicalDecoder,
    hierarchical_vfe,
    nll
)
from dataloader import SequenceData
from train import train, infer

_eps = 1e-20


def init(data, params):
    # Check if a set of covariance matrices has been specified
    covariances = None
    if params.cov_fn is not None:
        cov_path = os.path.join(params.data_dir, params.cov_fn)
        assert os.path.exists(cov_path), '{} does not exist'.format(
            cov_path)
        # if Matlab path, then load into matlab
        if os.path.splitext(cov_path)[1] == '.mat':
            import scipy.io as sio
            covariances = sio.loadmat(cov_path)['cov']
        else:
            covariances = np.load(cov_path)

    # Get the initial estimates of the covariance matrices
    params.W_init, params.B_init = fit_covariances(
        data, params.n_latent[-1], covariances, params.use_pca_cov_model,
        params.n_pc)

    encoder = HierarchicalEncoder(params)
    decoder = HierarchicalDecoder(params)

    if not params.use_pca_cov_model:
        decoder.W.requires_grad = False

    # Look to see if a checkpoint has already been specified
    if params.encoder_fn is not None:
        encoder_path = os.path.join(params.checkpoint_dir, params.encoder_fn)
        assert os.path.exists(encoder_path), '{} does not exist'.format(encoder_path)
        # Load the checkpoint
        if params.cuda:
            encoder.load_state_dict(torch.load(encoder_path))
        else:
            encoder.load_state_dict(torch.load(encoder_path, map_location={'cuda:0': 'cpu'}))

    if params.decoder_fn is not None:
        decoder_path = os.path.join(params.checkpoint_dir, params.decoder_fn)
        assert os.path.exists(decoder_path), '{} does not exist'.format(decoder_path)
        # Load the checkpoint
        # Load the checkpoint
        if params.cuda:
            decoder.load_state_dict(torch.load(decoder_path))
        else:
            decoder.load_state_dict(torch.load(decoder_path, map_location={
                'cuda:0': 'cpu'}))

    return encoder, decoder


def main(params):

    # Check to see if a checkpoint path has been specified
    if params.encoder_fn is not None and params.decoder_fn is not None:
        # Need to save some parameters so we don't overwrite them
        do_train = params.train
        do_infer = params.infer
        do_decode = params.decode
        epochs = params.num_epochs
        encoder_fn = params.encoder_fn
        decoder_fn = params.decoder_fn

        # Load params to ensure that we're using the same params
        with open(os.path.join(params.result_dir, 'params.json')) as f:
            params = json.load(f)

        # Convert params dictionary to an object
        class ParamDict2Obj(object):
            def __init__(self, dictionary):
                for key in dictionary:
                    setattr(self, key, dictionary[key])

        # Restore the saved parameters
        params = ParamDict2Obj(params)
        params.train = do_train
        params.infer = do_infer
        params.decode = do_decode
        params.num_epochs = epochs
        params.encoder_fn = encoder_fn
        params.decoder_fn = decoder_fn
    else:
        if not os.path.exists(params.result_dir):
            os.makedirs(params.result_dir)

        # Save model parameters
        with open(os.path.join(params.result_dir, 'params.json'), 'w') as outfile:
            json.dump(params.__dict__, outfile)

    np.random.seed(params.rand_seed)
    torch.manual_seed(params.rand_seed)

    params.checkpoint_dir = os.path.join(params.result_dir, 'checkpoint')
    if not os.path.exists(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    train_dir = os.path.join(params.data_dir, 'train')
    train_fns = [os.path.join('train', fn) for fn in os.listdir(train_dir) if
                 fn[:2] != '__']
    assert len(train_fns) > 0, 'No training files exist'
    train_sequences = SequenceData(
        params.data_dir, filenames=train_fns, seq_length=params.seq_length)

    valid_dir = os.path.join(params.data_dir, 'valid')
    valid_fns = [os.path.join('valid', fn) for fn in os.listdir(valid_dir) if
                 fn[:2] != '__']
    assert len(valid_fns) > 0, 'No validation files exist'
    valid_sequences = SequenceData(
        params.data_dir, filenames=valid_fns, seq_length=params.seq_length)

    params.n_input = train_sequences.data_dim

    encoder, decoder = init(train_sequences._xs, params)

    if params.train:
        train(train_sequences, valid_sequences, encoder, decoder,
               hierarchical_vfe, params)

    if params.infer:
        test_dir = os.path.join(params.data_dir, 'test')
        test_fns = [os.path.join('test', fn) for fn in os.listdir(test_dir) if
                    fn[:2] != '__']
        assert len(test_fns) > 0, 'No evaluation files exist'
        test_sequences = SequenceData(params.data_dir, filenames=test_fns)
        infer(test_sequences, encoder, decoder, hierarchical_vfe, nll, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)

    dir_ = os.path.dirname(os.path.realpath(__file__))
    default_data = os.path.join(dir_, 'data')
    default_result = os.path.join(dir_, 'results', str(int(time.time())))

    parser.add_argument('--data-dir',           '-d', default=default_data)
    parser.add_argument('--data-fn',            '-f', default='data.npy')
    parser.add_argument('--encoder-fn',       '-enc', default=None)
    parser.add_argument('--decoder-fn',       '-dec', default=None)
    parser.add_argument('--cov-fn',             '-c', default=None)
    parser.add_argument('--result-dir',         '-r', default=default_result)
    parser.add_argument('--temp',               '-t', type=float,   default=0.5)
    parser.add_argument('--net-length',         '-n', type=int,     default=200)
    parser.add_argument('--seq-length',         '-s', type=int,     default=1000)
    parser.add_argument('--batch-size',         '-b', type=int,     default=50)
    parser.add_argument('--use-pca-cov-model',  '-p', action='store_true')
    parser.add_argument('--n-latent',           '-k', type=int,     nargs='+',
                        required=True)
    parser.add_argument('--n-pc',              '-pc', type=int,     default=0)
    parser.add_argument('--n-hidden',           '-h', type=int,     nargs='+',
                        required=True)
    parser.add_argument('--n-layers',           '-l', type=int,     default=1)
    parser.add_argument('--diag',              '-dg', type=float,   default=5.)
    parser.add_argument('--num-epochs',         '-e', type=int,     default=100)
    parser.add_argument('--checkpoint-every',  '-ck', type=int,     default=20)
    parser.add_argument('--valid-every',        '-v', type=int,     default=10)
    parser.add_argument('--print-every',       '-pe', type=int,     default=1)
    parser.add_argument('--lr',                '-lr', type=float,   default=1e-3)
    parser.add_argument('--rand-seed',         '-rs', type=int,     default=42)
    parser.add_argument('--mask',               '-m', action='store_true')
    parser.add_argument('--cuda',               '-x', action='store_true')
    parser.add_argument('--bidirectional',     '-bi', action='store_true')
    parser.add_argument('--train',             '-tr', action='store_true')
    parser.add_argument('--infer',              '-i', action='store_true')
    parser.add_argument('--decode',           '-dss', action='store_true')

    args = parser.parse_args()
    main(args)
