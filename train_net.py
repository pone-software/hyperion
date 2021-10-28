from argparse import ArgumentParser
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from hyperion.models.photon_arrival_time.net import PhotonArivalTimePars


def make_funnel(max_neurons, layer_count):
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


def train_net(conf, train_data, test_data, writer=None):
    g = torch.Generator()
    g.manual_seed(31337)
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
        2,
        7,
        dropout=conf["dropout"],
        final_activations=[F.softplus] * 6 + [nn.Identity()],
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
            inp = train[:, :2]
            out = train[:, 2:]
            pred = net(inp)

            loss = criterion(pred, out)
            loss = loss.sum()
            loss.backward()

            total_train_loss += loss.item() * train.shape[0]

            optimizer.step()

        total_train_loss /= train_data.shape[0]

        total_test_loss = 0
        for test in test_loader:
            net.eval()

            inp = test[:, :2]
            out = test[:, 2:]
            pred = net(inp)

            loss = criterion(pred, out)
            loss = loss.sum()

            total_test_loss += loss.item() * test.shape[0]

        total_test_loss /= test_data.shape[0]

        if writer is not None:
            writer.add_scalar("Loss/train", total_train_loss, epoch)
            writer.add_scalar("Loss/test", total_test_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step()

    net.eval()
    inp = test_data[:, :2]
    out = test_data[:, 2:]
    pred = net(inp)
    loss = criterion(pred, out)
    loss = loss.sum()

    hparam_dict = dict(conf)

    if writer is not None:
        writer.add_hparams(hparam_dict, {"hparam/accuracy": loss})
        writer.flush()
        writer.close()

    return net


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-i", help="arrival time fit parameters file", required=True, dest="infile"
    )

    args = parser.parse_args()

    fit_results = pickle.load(open(args.infile, "rb"))
    data = []
    for d in fit_results:
        data.append(list(d["input"]) + list(d["output"]))
    data = np.asarray(np.vstack(data).squeeze(), dtype=np.float32)
    data[:, [2, 3, 4]] = np.sort(data[:, [2, 3, 4]], axis=1)
    data[:, 1] = np.log10(data[:, 1])
    data[:, -1] = np.log10(data[:, -1])
    data[:, -2] = -np.log10(1 - data[:, -2])

    rstate = np.random.RandomState(0)
    indices = np.arange(len(data))
    rstate.shuffle(indices)
    data_shuff = data[indices]

    split = int(0.5 * len(data))

    torch.random.manual_seed(31337)

    train_data = torch.tensor(data_shuff[:split])
    test_data = torch.tensor(data_shuff[split:])

    max_neurons = 700

    layer_count = 3
    layers = make_funnel(max_neurons, layer_count)
    conf = {
        "epochs": 1000,
        "batch_size": 500,
        "lr": 0.01,
        "dropout": 0.5,
        "max_neurons": max_neurons,
        "layer_count": layer_count,
    }
    writer = SummaryWriter(
        f"/tmp/tensorboard/runs/{conf['layer_count']}_{conf['max_neurons']}_{conf['batch_size']}_{conf['lr']}_{conf['epochs']}"
    )
    net = train_net(conf, train_data, test_data, writer)
