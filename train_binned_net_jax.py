import functools
import pickle

import haiku as hk
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
from hyperion.models.photon_binned_amplitude.net import make_forward_fn


class SimpleDataset(object):
    """Simple Dataset subclass that returns a tuple (input, output)."""

    def __init__(self, inputs, outputs, sanitize=True):
        super(SimpleDataset, self).__init__()
        self._inputs = inputs
        self._outputs = outputs
        self._sanitize = sanitize

        if len(self._inputs) != len(self._outputs):
            raise ValueError("Inputs and outputs must have same length.")

        self._len = len(self._inputs)

    def __getitem__(self, idx):
        """Return tuple of input, output."""
        out = self._outputs[idx]

        if isinstance(idx, int):
            idx = [idx]

        if self._sanitize:
            mask = np.isfinite(out)
            out[~mask] = 0
            return (
                np.atleast_2d(self._inputs[idx]),
                np.atleast_2d(out),
                np.atleast_2d(mask),
            )

        return np.atleast_2d(self._inputs[idx]), np.atleast_2d(out)

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


def train_net(conf, train_data, test_data, writer, rng):

    train_loader = DataLoader(
        train_data,
        batch_size=conf["batch_size"],
        shuffle=True,
        # worker_init_fn=seed_worker,
        rng=rng,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=conf["batch_size"],
        shuffle=False,
        # worker_init_fn=seed_worker,
        rng=rng,
    )

    forward_fn = make_forward_fn(conf)

    net = hk.transform_with_state(forward_fn)
    key = hk.PRNGSequence(42)

    params, state = net.init(next(key), next(iter(train_loader)), is_training=True)
    avg_params = params

    schedule = optax.cosine_decay_schedule(
        conf["lr"], conf["epochs"] * train_loader.n_batches, alpha=0.0
    )

    opt = optax.adam(learning_rate=schedule)
    opt_state = opt.init(params)

    def loss(params, state, rng_key, batch, is_training):
        pred, _ = net.apply(params, state, rng_key, batch, is_training)
        target = batch[1]
        mask = batch[2]
        se = 0.5 * (pred - target) ** 2

        nonzero = jnp.sum(mask, axis=0)
        mse = (jnp.sum(jnp.where(mask, se, jnp.zeros_like(se)), axis=0) / nonzero).sum()

        # Regularization (smoothness)
        first_diff = jnp.diff(pred, axis=1)
        first_diff_n = (
            first_diff - jnp.mean(first_diff, axis=1)[:, np.newaxis]
        ) / jnp.std(first_diff, axis=1)[:, np.newaxis]
        roughness = ((jnp.diff(first_diff_n, axis=1) ** 2) / 4).sum()

        roughness_weight = 1

        return 1 / (roughness_weight + 1) * (mse + roughness_weight * roughness)

    @functools.partial(jax.jit, static_argnums=[5])
    def get_updates(params, state, rng_key, opt_state, batch, is_training):
        """Learning rule (stochastic gradient descent)."""
        l, grads = jax.value_and_grad(loss)(
            params, state, rng_key, batch, is_training=is_training
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return l, new_params, opt_state

    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    for epoch in range(conf["epochs"]):
        # Train/eval loop.
        train_loss = 0
        for train in train_loader:
            rng_key = next(key)
            l, params, opt_state = get_updates(
                params, state, rng_key, opt_state, train, is_training=True
            )
            avg_params = ema_update(params, avg_params)

            train_loss += l * len(train[0])
        train_loss /= len(train_data)

        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(
                test[0]
            )
        test_loss /= len(test_data)

        if writer is not None:
            train_loss, test_loss, lr = jax.device_get(
                (train_loss, test_loss, schedule(opt_state[1].count))
            )
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("LR", lr, epoch)

    @jax.jit
    def net_eval_fn(x):
        return net.apply(avg_params, state, None, x, is_training=False)[0]

    if writer is not None:
        test_loss = 0
        for test in test_loader:
            test_loss += loss(avg_params, state, None, test, is_training=False) * len(
                test[0]
            )
        test_loss /= len(test_data)

        hparam_dict = dict(conf)
        writer.add_hparams(hparam_dict, {"hparam/test_loss": np.asarray(test_loss)})
        writer.flush()
        writer.close()

    return net_eval_fn, avg_params, state


# for tts in [2, 3, 4]:
# for tts in [0]:
tts = 2
hists = pickle.load(open(f"data/combined_photon_hists_{tts}.pickle", "rb"))
inputs = np.asarray(hists[::2])
outputs = np.asarray(hists[1::2])
inputs = inputs.reshape((inputs.shape[0] * inputs.shape[1], 2))
outputs = outputs.reshape((outputs.shape[0] * outputs.shape[1], 529))

outputs = np.log(outputs)

first_finite = np.nonzero(
    np.sum(~np.isfinite(outputs), axis=0) / outputs.shape[0] < 0.5
)[0][0]
# print(first_finite)
outputs[:, :first_finite] = -300

binning = jnp.arange(-30, 500, 1)


inputs[:, 1] = jnp.log10(inputs[:, 1])
data = SimpleDataset(inputs, outputs)
split = int(0.5 * len(data))
rng = np.random.RandomState(2)
train_data, test_data = create_random_split(data, split, rng)


for n_neurons in [1000]:
    for epochs in [700, 900, 1000]:
        for lr in [0.007, 0.01, 0.02]:
            for dropout in [0, 0.2, 0.3]:
                conf = {
                    "batch_size": 500,
                    "n_in": 2,
                    "n_out": outputs.shape[1],
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "n_neurons": n_neurons,
                }
            ident_str = f"hist_{conf['batch_size']}_{conf['lr']}_{conf['epochs']}_{conf['dropout']}_{conf['n_neurons']}"
            writer = SummaryWriter(os.path.join("/tmp/tensorboard/runs/", ident_str))

            net, params, state = train_net(conf, train_data, test_data, writer, rng)
            pickle.dump([params, state], open(f"data/{ident_str}.pickle", "wb"))


"""

conf = {
    "batch_size": 400,
    "n_in": 2,
    "n_out": outputs.shape[1],
    "dropout": 0,
    "lr": 0.001,
    "epochs": 500,
    "n_neurons": 1500,
}
writer = SummaryWriter(
    f"/tmp/tensorboard/runs/hist_test_{conf['batch_size']}_{conf['lr']}_{conf['epochs']}_{conf['dropout']}_{conf['n_neurons']}"
)


net, params, state = train_net(conf, train_data, test_data, writer, rng)
"""
pickle.dump([params, state, conf], open("data/arrival_hist_net_2tts_jax.pickle", "wb"))

params, state, conf = pickle.load(open("data/arrival_hist_net_2tts_jax.pickle", "rb"))
forward_fn = make_forward_fn(conf)
net = hk.transform_with_state(forward_fn)


@jax.jit
def net_eval_fn(x):
    return net.apply(params, state, None, x, is_training=False)[0]
