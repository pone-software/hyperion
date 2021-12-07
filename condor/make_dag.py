import numpy as np
from scipy.stats import qmc

sampler = qmc.Sobol(d=1, scramble=False)
sample = sampler.random_base2(m=7)
dists = (10 ** qmc.scale(sample, np.log10(5), np.log10(300))).squeeze()


with open("generate_photons.dag", "w") as hdl:
    for i, dist in enumerate(dists):
        hdl.write(f"JOB {i}_photons submit_photons.sub\n")
        hdl.write(
            f'VARS {i}_photons dist="{dist}" outfile="photon_table_{i}.pickle" seed="{i}"\n'
        )

        hdl.write(f"JOB {i}_fit submit_fit.sub\n")
        hdl.write(
            f'VARS {i}_fit infile="photon_table_{i}.pickle" outfile="photon_fitpars_{i}.pickle"\n'
        )
        hdl.write(f"PARENT {i}_photons CHILD {i}_fit\n")

with open("generate_photons_second.dag", "w") as hdl:
    for i, dist in enumerate(dists):
        hdl.write(f"JOB {i}_fit submit_fit.sub\n")
        hdl.write(
            f'VARS {i}_fit infile="photon_table_{i}.pickle" outfile="photon_fitpars_{i}.pickle" seed="{i}"\n'
        )

with open("generate_photons_third.dag", "w") as hdl:
    for i, dist in enumerate(dists):
        for tts in [0, 2, 3, 4]:
            hdl.write(f"JOB {i}_hist_{tts} submit_hists.sub\n")
            hdl.write(
                f'VARS {i}_hist_{tts} infile="photon_table_{i}.pickle" outfile="photon_hists_{i}_{tts}.pickle" seed="{i}" tts="{tts}"\n'
            )
