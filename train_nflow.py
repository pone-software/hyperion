import os
import pickle

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from hyperion.constants import Constants
from hyperion.data import DataLoader, SimpleDataset, create_random_split
from hyperion.models.photon_arrival_time_nflow.net import (
    train_shape_model,
    train_counts_model,
)


data = np.load("data/norm_flow_dset_1.45tts.npz")["times"]
# data = pickle.load(open("data/photon_nflow_ds_1_1.45.pickle", "rb"))

dset = SimpleDataset(np.log10(data[1, :]), data[2, :], data[0, :])

rng = np.random.RandomState(2)
# dset = downsample_ds(dset, 0.1, rng, False)
split = int(0.9 * len(dset))

train_data, test_data = create_random_split(dset, split, rng)
print(len(test_data))

config = {
    "flow_num_layers": 2,
    "flow_num_bins": 10,
    "flow_rmin": 0,
    "flow_rmax": 500,
    "mlp_hidden_size": 400,
    "mlp_num_layers": 3,
    "lr": 0.001,
    "batch_size": 10000,
    "steps": 4000,
}

train_loader = DataLoader(
    train_data,
    batch_size=config["batch_size"],
    infinite=True,
    shuffle=True,
    rng=rng,
)
test_loader = DataLoader(
    test_data,
    batch_size=config["batch_size"],
    shuffle=False,
    infinite=False,
    rng=rng,
)

ident_str = (
    f"{config['flow_num_layers']}_{config['flow_num_bins']}_{config['mlp_hidden_size']}"
    + f"_{config['mlp_num_layers']}_{config['batch_size']}_{config['steps']}_{config['lr']}"
)

writer = SummaryWriter(f"/tmp/tensorboard/runs/norm_flow_cos_anneal_{ident_str}")

params = train_shape_model(config, train_loader, test_loader, writer=writer)

pickle.dump(
    (config, params), open("data/photon_arrival_time_nflow_params.pickle", "wb")
)


counts = np.load("data/norm_flow_dset_1.45tts.npz", allow_pickle=True)["counts"]
counts[1] = np.concatenate(counts[1])
counts = np.asarray(counts, dtype=np.float64)

dset = SimpleDataset(np.log10(counts[0, :]), counts[1, :], np.log10(counts[2, :]))
rng = np.random.RandomState(2)

split = int(0.9 * len(dset))

train_data, test_data = create_random_split(dset, split, rng)

config = {
    "mlp_hidden_size": 700,
    "mlp_num_layers": 3,
    "lr": 0.05,
    "batch_size": 200,
    "steps": 30000,
}

train_loader = DataLoader(
    train_data,
    batch_size=config["batch_size"],
    infinite=True,
    shuffle=True,
    rng=rng,
)
test_loader = DataLoader(
    test_data,
    batch_size=config["batch_size"],
    shuffle=False,
    infinite=False,
    rng=rng,
)

ident_str = (
    f"{config['mlp_hidden_size']}"
    + f"_{config['mlp_num_layers']}_{config['batch_size']}_{config['steps']}_{config['lr']}"
)

writer = SummaryWriter(f"/tmp/tensorboard/runs/counts_cos_anneal_{ident_str}")

params = train_counts_model(config, train_loader, test_loader, writer=writer)

pickle.dump(
    (config, params), open("data/photon_arrival_time_counts_params.pickle", "wb")
)
