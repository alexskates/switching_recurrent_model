import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os


class SequenceData(Dataset):

    def __init__(self, folder_dataset, filenames=['data.npy'],
                 seq_length=0, transform=None):
        self._xs_all = []
        self._xs = []

        self.transform = transform
        self.seq_length = seq_length
        self.filenames = filenames

        # Load the raw data
        for fn in self.filenames:
            tmp = np.load(os.path.join(folder_dataset, fn)).astype(np.float32)
            if len(tmp.shape) > 2:
                for x in tmp:
                    if x.shape[1] < x.shape[0]:
                        self._xs_all.append(x)
                    else:
                        self._xs_all.append(x.T)
            else:
                if tmp.shape[1] < tmp.shape[0]:
                    self._xs_all.append(tmp)
                else:
                    self._xs_all.append(tmp.T)

        # If seq_length=0, then use the length of the smallest sequence
        if self.seq_length == 0:
            self.seq_length = min([x.shape[0] for x in self._xs_all])

        # Split the data up into "independent" sequences of length length
        for x in self._xs_all:
            self._xs.append(
                x[:self.seq_length * (x.shape[0] // self.seq_length)].reshape(
                    int(x.shape[0] / self.seq_length),
                    self.seq_length,
                    -1))
        # Join along 0 dimension
        self._xs = np.concatenate(self._xs, 0)

        self.data_dim = self._xs.shape[-1]
        self.data_len = min([x.shape[0] for x in self._xs_all])

    # Give pytorch access to sequences
    def __getitem__(self, index):
        seq = self._xs[index]
        if self.transform is not None:
            seq = self.transform(seq)

        # Convert to tensors
        return torch.from_numpy(seq)

    # Override to get size of dataset
    def __len__(self):
        return self._xs.shape[0]
