from argparse import ArgumentParser
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hyperion.models.photon_arrival_time.net import PhotonArivalTimePars


class SimpleDataset(Dataset):
    """Simple Dataset subclass that returns a tuple (input, output)."""

    def __init__(self, inputs, outputs):
        super(SimpleDataset, self).__init__()
        self._inputs = inputs
        self._outputs = outputs

        if len(self._inputs) != len(self._outputs):
            raise ValueError("Inputs and outputs must have same length.")

        self._len = len(self._inputs)

    def __getitem__(self, idx):
        """Return tuple of input, output."""
        return self._inputs[idx], self._outputs[idx]

    def __len__(self):
        return self._len


def make_funnel(max_neurons, layer_count):
    """Create a neuron per layer list for a funnel shape."""
    layers = []
    out_feat = 7
    previous = max_neurons
    layers.append(max_neurons)
    step_size = int((previous - out_feat) / (layer_count))
    step_size = max(0, step_size)
    for _ in range(layer_count - 1):
        previous = previous - step_size
        layers.append(previous)
    return layers


def train_param_net(conf, train_data, test_data, writer=None, seed=31337):
    """Train a funnel shaped MLP."""

    g = torch.Generator()
    torch.random.manual_seed(seed)
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_data,
        batch_size=conf["batch_size"],
        shuffle=True,
        # worker_init_fn=seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=conf["batch_size"],
        shuffle=False,
        # worker_init_fn=seed_worker,
        generator=g,
    )

    layers = make_funnel(conf["max_neurons"], conf["layer_count"])
    net = PhotonArivalTimePars(
        layers,
        conf["n_in"],
        conf["n_out"],
        dropout=conf["dropout"],
        final_activations=conf["final_activations"],
    )
    optimizer = optim.Adam(net.parameters(), lr=conf["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, conf["epochs"])

    def criterion(pred, target):
        # print(pred.shape, target.shape)
        mse = torch.mean((pred - target) ** 2, axis=0)
        return mse

    for epoch in range(conf["epochs"]):
        total_train_loss = 0
        for train in train_loader:
            net.train()
            optimizer.zero_grad()
            inp, out = train
            pred = net(inp)

            loss = criterion(pred, out)
            loss = loss.sum()
            loss.backward()

            total_train_loss += loss.item() * inp.shape[0]

            optimizer.step()

        total_train_loss /= len(train_data)

        total_test_loss = 0
        for test in test_loader:
            net.eval()

            inp, out = test
            pred = net(inp)

            loss = criterion(pred, out)
            loss = loss.sum()

            total_test_loss += loss.item() * inp.shape[0]

        total_test_loss /= len(test_data)

        if writer is not None:
            writer.add_scalar("Loss/train", total_train_loss, epoch)
            writer.add_scalar("Loss/test", total_test_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step()

    """
    net.eval()
    inp, out = test_data[:]
    pred = net(inp)
    loss = criterion(pred, out)
    loss = loss.sum()

    hparam_dict = dict(conf)

    if writer is not None:
        writer.add_hparams(hparam_dict, {"hparam/accuracy": loss})
        writer.flush()
        writer.close()
    """

    return net


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i", help="arrival time fit parameters file", required=True, dest="infile"
    )
    parser.add_argument("-o", help="model output file", required=True, dest="outfile")

    args = parser.parse_args()

    fit_results = pickle.load(open(args.infile, "rb"))

    data = []
    for d in fit_results:
        data.append(
            list(d["input"]) + list(d["output_tres"]) + list(d["output_arrv_pos"])
        )
    data = np.asarray(np.vstack(data).squeeze(), dtype=np.float32)
    data[:, [2, 3, 4]] = np.sort(data[:, [2, 3, 4]], axis=1)
    data[:, 1] = np.log10(data[:, 1])
    data[:, 8] = np.log10(data[:, 8])
    data[:, 7] = -np.log10(1 - data[:, 7])

    rstate = np.random.RandomState(0)
    indices = np.arange(len(data))
    rstate.shuffle(indices)
    columns = list(range(9)) + [12, 13]

    data_shuff = data[indices][:, columns]

    split = int(0.5 * len(data))

    torch.random.manual_seed(31337)

    train_data = torch.tensor(data_shuff[:split])
    test_data = torch.tensor(data_shuff[split:])

    train_dataset = SimpleDataset(train_data[:, :2], train_data[:, 2:])
    test_dataset = SimpleDataset(test_data[:, :2], test_data[:, 2:])

    max_neurons = 800
    layer_count = 3

    conf = {
        "epochs": 1000,
        "batch_size": 400,
        "lr": 0.01,
        "dropout": 0.3,
        "max_neurons": max_neurons,
        "layer_count": layer_count,
        "n_in": 2,
        "n_out": train_data.shape[1],
        "final_activations": [F.softplus] * 6 + [nn.Identity()] + [F.softplus] * 2,
    }

    writer = SummaryWriter(
        f"/tmp/tensorboard/runs/{conf['layer_count']}_{conf['max_neurons']}_{conf['batch_size']}_{conf['lr']}_{conf['epochs']}_{conf['dropout']}"
    )
    net = train_param_net(conf, train_dataset, test_dataset, writer)
    torch.save(net, args.outfile)
