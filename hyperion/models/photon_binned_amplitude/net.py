import haiku as hk
import jax
import jax.numpy as jnp
import pickle


class HistMLP(hk.Module):
    def __init__(self, output_size, layers, dropout, final_activations, name=None):
        super().__init__(name=name)
        self.output_size = output_size
        self.layers = layers
        self.dropout = dropout
        self.final_activations = final_activations

    def __call__(self, x, is_training):
        for n_per_layer in self.layers:
            x = hk.Linear(n_per_layer)(x)
            # x = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(x, is_training=is_training)
            x = jax.nn.relu(x)
            if is_training:
                key = hk.next_rng_key()
                x = hk.dropout(key, self.dropout, x)

        x = hk.Linear(self.output_size)(x)

        if self.final_activations is not None:
            x = self.final_activations(x)

        return x


def make_forward_fn(conf):
    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(batch, is_training):
        inp = jnp.asarray(batch[0], dtype=jnp.float32)
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, is_training)

    return forward_fn


def make_eval_forward_fn(conf):
    layers = [conf["n_neurons"], conf["n_neurons"], conf["n_neurons"]]

    def forward_fn(inp):
        return HistMLP(conf["n_out"], layers, conf["dropout"], None)(inp, False)

    return forward_fn


def make_net_eval_from_pickle(path):
    params, state, conf, binning = pickle.load(open(path, "rb"))
    forward_fn = make_eval_forward_fn(conf)
    net = hk.transform_with_state(forward_fn)

    @jax.jit
    def net_eval_fn(x):
        return net.apply(params, state, None, x)[0]

    return net_eval_fn, binning
