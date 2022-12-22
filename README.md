# Continuous Patient State Attention Models

Code for the papers:
> [Vinod Kumar Chauhan, Anshul Thakur, Odhran O'Donoghue, Omid Rohanian and David A. Clifton (2023) Continuous Patient State Attention Models, (under review)](https://arxiv.org/abs/2208.03196)

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
@inproceedings{chauhan2022coper,
  title={Continuous Patient State Attention Models},
  author={Chauhan, Vinod Kumar and Thakur, Anshul and O'Donoghue, Odhran and Rohanian, Omid and Clifton, David A},
  year={2023},
}

@inproceedings{chauhan2022coper,
  title={COPER: Continuous Patient State Perceiver},
  author={Chauhan, Vinod Kumar and Thakur, Anshul and O'Donoghue, Odhran and Clifton, David A},
  booktitle={IEEE International Conference on Biomedical and Health Informatics},
  year={2022},
  url={https://arxiv.org/abs/2208.03196}
}
```

Neural ODE implementations based on [[Yulia Rubanova]](https://github.com/YuliaRubanova/latent_ode).
