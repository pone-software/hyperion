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
from hyperion.models.photon_binned_amplitude.net import (
    make_forward_fn,
    train_net,
    make_logp1_trafo,
)
from hyperion.data import SimpleDataset, create_random_split

tts = 2
hists = pickle.load(open(f"data/combined_photon_hists_{tts}.pickle", "rb"))
inputs = np.asarray(hists[::2])
outputs = np.asarray(hists[1::2])
inputs = inputs.reshape((inputs.shape[0] * inputs.shape[1], 2))
outputs = outputs.reshape((outputs.shape[0] * outputs.shape[1], outputs.shape[-1]))

trafo_scale = 1e18
trafo, rev_trafo = make_logp1_trafo(trafo_scale)

outputs = trafo(outputs)

binning = jnp.concatenate(
    [jnp.linspace(0, 10, 11), jnp.logspace(1, jnp.log10(500), 25)[1:]]
)


inputs[:, 1] = jnp.log10(inputs[:, 1])
data = SimpleDataset(inputs, outputs)
split = int(0.5 * len(data))
rng = np.random.RandomState(2)
train_data, test_data = create_random_split(data, split, rng)


for n_neurons in [300, 500, 1500]:
    for epochs in [200, 400, 700]:
        for lr in [0.001, 0.005, 0.01]:
            for dropout in [0, 0.2, 0.3]:
                conf = {
                    "batch_size": 500,
                    "n_in": 2,
                    "n_out": outputs.shape[1],
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "n_neurons": n_neurons,
                    "final_activations": jax.nn.softplus,
                }
                ident_str = f"hist_{conf['batch_size']}_{conf['lr']}_{conf['epochs']}_{conf['dropout']}_{conf['n_neurons']}"
                writer = SummaryWriter(
                    os.path.join("/tmp/tensorboard/runs/", ident_str)
                )

                net, params, state = train_net(conf, train_data, test_data, writer, rng)
                pickle.dump(
                    [params, state, conf, binning, trafo_scale],
                    open(f"data/{ident_str}.pickle", "wb"),
                )


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

pickle.dump([params, state, conf], open("data/arrival_hist_net_2tts_jax.pickle", "wb"))

params, state, conf = pickle.load(open("data/arrival_hist_net_2tts_jax.pickle", "rb"))
forward_fn = make_forward_fn(conf)
net = hk.transform_with_state(forward_fn)


@jax.jit
def net_eval_fn(x):
    return net.apply(params, state, None, x, is_training=False)[0]
"""
