import numpy as np


class SimpleDataset(object):
    """Simple Dataset subclass that returns a tuple (input, output)."""

    def __init__(self, *arrays):
        super(SimpleDataset, self).__init__()
        self._arrays = arrays

        self._len = len(arrays[0])

        for arr in arrays:
            if len(arr) != self._len:
                raise ValueError("Inputs and outputs must have same length.")

    def __getitem__(self, idx):
        """Return tuple of input, output."""

        if isinstance(idx, int):
            idx = [idx]

        outs = [np.atleast_2d(arr[idx]) for arr in self._arrays]

        return outs

    def __len__(self):
        return self._len


class SubSet(object):
    """Dataset subset."""

    def __init__(self, dataset, subset_ix):
        super(SubSet, self).__init__()
        if max(subset_ix) > len(dataset):
            raise RuntimeError("Invalid index")

        self._subset_ix = subset_ix
        self._len = len(subset_ix)
        self._dataset = dataset

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        true_ix = self._subset_ix[idx]
        return self._dataset[true_ix]


def create_random_split(dataset, split_len, rng):
    """Create a random split."""
    ixs = np.arange(len(dataset))
    rng.shuffle(ixs)
    first_split = SubSet(dataset, ixs[:split_len])
    second_split = SubSet(dataset, ixs[split_len:])

    return first_split, second_split


def randomize_ds(dataset, rng):
    """Randomize a dataset."""
    ixs = np.arange(len(dataset))
    rng.shuffle(ixs)
    return SubSet(dataset, ixs)


class DataLoader(object):
    """Dataloader"""

    def __init__(self, dataset, batch_size, rng, shuffle=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._rng = rng
        self._shuffle = shuffle
        self._n_batches = int(np.ceil(len(self._dataset) / self._batch_size))

    def __iter__(self):
        if self._shuffle:
            ds = randomize_ds(self._dataset, self._rng)
        else:
            ds = self._dataset

        for batch in range(self._n_batches):
            upper = min(len(ds), (batch + 1) * self._batch_size)
            ixs = np.arange(batch * self._batch_size, upper)
            yield ds[ixs]

    @property
    def n_batches(self):
        return self._n_batches
