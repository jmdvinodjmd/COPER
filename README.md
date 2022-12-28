# Continuous Patient State Attention Models

Code for the papers:
> [Vinod Kumar Chauhan, Anshul Thakur, Odhran O'Donoghue, Omid Rohanian and David A. Clifton (2023) Continuous Patient State Attention Models, (under review)](https://www.medrxiv.org/content/10.1101/2022.12.23.22283908v1)

> [Vinod Kumar Chauhan, Anshul Thakur, Odhran O'Donoghue and David A. Clifton (2022) COPER: Continuous Patient State Perceiver, IEEE International Conference on Biomedical and Health Informatics (BHI-2022)](https://arxiv.org/abs/2208.03196)

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Experiment

You need mimic-iii [mimic-iii](https://github.com/YerevaNN/mimic3-benchmarks) and [Physionet Challenge 2012](https://physionet.org/content/challenge-2012/1.0.0/) datasets.

To run different experiments, you can use the following shell scripts:
* ```run_irregular_mimic.sh```: to study irregularity on mimic dataset.
* ```run_irregular_physionet.sh```: to study irregularity on physionet dataset.
* ```run_exp_normal.sh```: to run Perceiver and baselines (without irregularity).
* ```run_perceiver_latents.sh```: to study the effect of number of latents on the Perceiver and compare with Transformer.

## Citations:
```
@article {Chauhan2022b,
	author = {Chauhan, Vinod K. and Thakur, Anshul and O{\textquoteright}Donoghue, Odhran and Rohanian, Omid and Clifton, David A.},
	title = {Continuous Patient State Attention Models},
	year = {2022},
	doi = {10.1101/2022.12.23.22283908},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2022/12/26/2022.12.23.22283908},
	journal = {medRxiv}
}

@inproceedings{Chauhan2022a,
  title={COPER: Continuous Patient State Perceiver},
  author={Chauhan, Vinod Kumar and Thakur, Anshul and O'Donoghue, Odhran and Clifton, David A},
  booktitle={IEEE International Conference on Biomedical and Health Informatics},
  year={2022},
  url={https://arxiv.org/abs/2208.03196}
}
```

Neural ODE implementations based on [[Yulia Rubanova]](https://github.com/YuliaRubanova/latent_ode).
