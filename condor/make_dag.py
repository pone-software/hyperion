import numpy as np
from scipy.stats import qmc

sampler = qmc.Sobol(d=1, scramble=False)
sample = sampler.random_base2(m=7)
dists = (10 ** qmc.scale(sample, 0, np.log10(500))).squeeze()

with open("generate_photons.dag", "w") as hdl:
    for i, dist in enumerate(dists):
        hdl.write(f"JOB {i} submit.sub\n")
        hdl.write(f"VARS {i} dist=\"{dist}\" outfile=\"photon_table_{i}.pickle\"\n")

