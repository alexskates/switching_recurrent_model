import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
from os.path import join
import datetime


def as_minutes(timedelta):
    return (timedelta.seconds // 60)


def time_since(since, percent):
    now = datetime.datetime.now()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def run_epoch(data_loader, encoder, decoder, encoder_optimiser,
          decoder_optimiser, loss_fn, params, backward=True):

    if backward:
        encoder_optimiser.zero_grad()
        decoder_optimiser.zero_grad()

    temp = torch.Tensor([params.temp])
    if params.cuda:
        temp = temp.cuda()

    losses = []
    lvs = []

    for x_seq in iter(data_loader):
        h = encoder.init_hidden(x_seq.shape[0])
        h_0 = decoder.init_hidden(x_seq.shape[0])

        loss = 0.

        for x in x_seq.split(params.net_length, 1):

            if h is not None:
                if type(h) == list:
                    h = [Variable(h_element.data) for h_element in h]
                else:
                    h = Variable(h.data)
            if h_0 is not None:
                if type(h_0) == list:
                    h_0 = [Variable(h_0_element.data) for h_0_element in h_0]
                else:
                    h_0 = Variable(h_0.data)

            if params.cuda:
                x = x.cuda()
                if h is not None:
                    if type(h) == list:
                        h = [h_element.cuda() for h_element in h]
                    else:
                        h = h.cuda()
                if h_0 is not None:
                    if type(h_0) == list:
                        h_0 = [h_0_element.cuda() for h_0_element in h_0]
                    else:
                        h_0 = h_0.cuda()

            logits_z, q, h = encoder(x, h)
            c, E, p, h_0 = decoder(temp, logits_z, h_0)

            loss = loss_fn(x, E, c, q, p).sum()
            losses.append(loss.item() / x.shape[0])

            if 'alpha' in q:
                lvs.append(q['alpha'][1].data.cpu().numpy())

            if backward:
                loss.backward()

                # nn.utils.clip_grad_norm_(encoder.parameters(), 10)
                # nn.utils.clip_grad_norm_(decoder.parameters(), 10)

                encoder_optimiser.step()
                decoder_optimiser.step()

    return losses, lvs


def train(train_sequence, valid_sequence, encoder, decoder, loss_fn, params):
    start = datetime.datetime.now()

    train_losses = []
    valid_losses = []
    lvs_all = []

    train_loader = DataLoader(train_sequence, batch_size=params.batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_sequence, batch_size=params.batch_size,
                              shuffle=True)

    print(
        'Training\n'
        '{:%d, %b %Y %H:%M}\n'
        'Data files: \n {} \n'
        '------------\n\n'.format(
            start,
            '\n'.join(
                [join(params.data_dir, fn) for fn in
                 train_sequence.filenames]))
    )

    encoder_optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, encoder.parameters()), lr=params.lr)
    decoder_optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, decoder.parameters()), lr=params.lr)

    try:
        for i in range(params.num_epochs):
            train_loss, lvs = run_epoch(data_loader=train_loader,
                                   encoder=encoder,
                                   decoder=decoder,
                                   encoder_optimiser=encoder_optimiser,
                                   decoder_optimiser=decoder_optimiser,
                                   loss_fn=loss_fn,
                                   params=params,
                                   backward=True)
            train_losses.append(np.mean(train_loss))
            lvs_all.extend(lvs)

            if i % params.valid_every == 0:
                valid_loss, _ = run_epoch(data_loader=valid_loader,
                                       encoder=encoder,
                                       decoder=decoder,
                                       encoder_optimiser=None,
                                       decoder_optimiser=None,
                                       loss_fn=loss_fn,
                                       params=params,
                                       backward=False)
                valid_losses.append(np.mean(valid_loss))

                if len(valid_losses) > 1:
                    plt.plot(np.arange(0, i+1), train_losses, label='Training',
                             c='blue')
                    plt.plot(np.arange(0, i+1, params.valid_every),
                             valid_losses, label='Validation', c='orange',
                             linestyle='--')
                    plt.legend()
                    plt.xlabel('Epochs')
                    plt.ylabel('VFE')
                    plt.savefig(join(params.result_dir, 'training.png'))
                    plt.close()

            if i % params.checkpoint_every == 0:
                enc_path = join(params.checkpoint_dir, 'encoder_{}.pt'.format(i))
                dec_path = join(params.checkpoint_dir, 'decoder_{}.pt'.format(i))
                torch.save(encoder.state_dict(), enc_path)
                torch.save(decoder.state_dict(), dec_path)

            if i % params.print_every == 0:
                print('%s (%d/%d %d%%) %.4f' % (
                    time_since(start, (i+1) / params.num_epochs),
                    (i + 1), params.num_epochs, (i+1) / params.num_epochs * 100,
                    np.mean(train_loss)))

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(e)
    finally:
        torch.save(encoder.state_dict(),
                   join(params.checkpoint_dir, 'encoder_final.pt'))
        torch.save(decoder.state_dict(),
                   join(params.checkpoint_dir, 'decoder_final.pt'))

        np.save(join(params.result_dir, 'train_loss.npy'), np.array(train_losses))
        np.save(join(params.result_dir, 'valid_loss.npy'), np.array(valid_losses))

    print('Training complete, final loss: {}, time taken: {}\n'.format(
        train_losses[-1], str(datetime.datetime.now() - start)))


def infer(eval_sequence, encoder, decoder, loss_fn, nll_fn, params):
    start = datetime.datetime.now()

    # If dropout is being used, turn off
    encoder.eval()
    decoder.eval()

    eval_losses = []

    # As we're using the history, want to just feed in the sequences in order
    eval_loader = DataLoader(eval_sequence, batch_size=len(eval_sequence),
                             shuffle=False)

    q_sequence = {
        param: np.zeros(
            shape=(len(eval_sequence), eval_sequence.seq_length, latent_dim))
        for (param, latent_dim) in encoder.distributions.items()
    }

    vfe = np.empty(shape=(len(eval_sequence), eval_sequence.seq_length))
    nll = np.empty(shape=(len(eval_sequence), eval_sequence.seq_length))

    temp = torch.Tensor([params.temp])
    if params.cuda:
        temp = temp.cuda()

    print(
        'Inference\n'
        '{:%d, %b %Y %H:%M}\n'
        'Data files: \n {} \n'
        '------------\n\n'.format(
            start,
            '\n'.join(
                [join(params.data_dir, fn) for fn in
                 eval_sequence.filenames]))
    )

    for b, x_seq in enumerate(iter(eval_loader)):
        h = encoder.init_hidden(x_seq.shape[0])
        h_0 = decoder.init_hidden(x_seq.shape[0])

        for i, x in enumerate(x_seq.split(params.net_length, 1)):

            if h is not None: h = Variable(h.data)
            if h_0 is not None: h_0 = Variable(h_0.data)
            x = Variable(x, requires_grad=True)

            if params.cuda:
                x = x.cuda()
                if h is not None: h = h.cuda()
                if h_0 is not None: h_0 = h_0.cuda()

            logits, q, h = encoder(x, h)
            c, E, p, h_0 = decoder(temp, logits, h_0)

            loss_batch = loss_fn(x, E, c, q, p)
            nll_batch = nll_fn(x, E, c)

            eval_losses.append(loss_batch.sum().item() / x.shape[0])

            idx_start = b * params.seq_length + i * params.net_length
            idx_end = b * params.seq_length + (i + 1) * params.net_length
            for (param, inf) in q.items():
                if type(inf) is tuple and param == 'alpha':
                    q_sequence[param][:, idx_start:idx_end] = inf[0].data.cpu().numpy()
                    q_alpha_lv = inf[1].data.cpu().numpy()
                else:
                    q_sequence[param][:, idx_start:idx_end] = inf.data.cpu().numpy()
            vfe[:, idx_start:idx_end] = loss_batch.data.cpu().numpy()
            nll[:, idx_start:idx_end] = nll_batch.data.cpu().numpy()

    print('Inference complete, time taken: {}\n'.format(
        str(datetime.datetime.now() - start)))

    # Save everything
    for (param, seq) in q_sequence.items():
        if param == 'alpha':
            np.save(join(params.result_dir, 'q_alpha_lv.npy'), q_alpha_lv)
        np.save(join(params.result_dir, 'q_%s_all.npy' % param), seq)
    np.save(join(params.result_dir, 'vfe_all.npy'), vfe)
    np.save(join(params.result_dir, 'nll_all.npy'), nll)
    np.save(join(params.result_dir, 'inferred_covariances.npy'), c.data.cpu().numpy())
    np.save(join(params.result_dir, 'eval_loss.npy'), np.array(eval_losses))
