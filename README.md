![BWLer: Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for PINNs](assets/banner.png)

# BWLer: Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for PINNs

This repository contains code for the following paper:

> **BWLer: Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for PINNs.**
>
> Jerry Liu, Yasa Baig, Denise Hui Jean Lee, Rajat Vadiraj Dwaraknath, Atri Rudra, Chris Ré.
> 
> Workshop on the Theory of AI for Scientific Computing (TASC) @ COLT 2025 [Best Paper Award].  
> [[Paper]](https://openreview.net/forum?id=rKZkotB0um)

![Standard PINN evaluates an MLP throughout the domain (left). BWLer interpolates globally based on values at discrete grid nodes; BWLERhatted MLP obtains values using an MLP (middle), explicit BWLer parameterizes values directly (right).](assets/method.png)


## Dependencies
Install dependencies with
```
conda create -n "bwler" python=3.11
conda activate bwler
pip install -e .
```


## Code structure
The code is organized as follows:
- [scripts/](scripts/): contains scripts for running the experiments:
  - [scripts/pdes/](scripts/pdes/): PDE experiment submission scripts
  - [scripts/ablations/](scripts/ablations/): ablation study scripts
- [src/experiments/](src/experiments/): contains the main experiment framework, including the PDE problem definitions
- [src/models/](src/models/): contains the two BWLer variants:
  - [src/models/interpolant_nd.py](src/models/interpolant_nd.py): explicit BWLer
  - [src/models/mlp_interpolant_nd.py](src/models/mlp_interpolant_nd.py): BWLer-hatted MLP
  - [src/models/mlp.py](src/models/mlp.py): standard MLP
- [src/optimizers/](src/optimizers/): contains the Nyström-Newton CG optimizer


## Getting started

- To try BWLer on the five benchmark PDEs from our paper, run the scripts in [scripts/pdes/](scripts/pdes/).
- To incorporate new PDE problems into the repo, create a new class extending [base_pde.py](src/experiments/pdes/base_pde.py). Simply specify the domain in the `__init__` and PDE loss terms in `get_loss_dict`. Please refer to [convection.py](src/experiments/pdes/simple/convection.py) for a simple example, and [poisson_2d_cg.py](src/experiments/pdes/benchmarks/poisson_2d_cg.py) for an example with an irregular domain.
- To try different optimizers or training techniques, refer to [base_fcn.py](src/experiments/base_fcn.py) for the optimizer initialization and [base_pde.py](src/experiments/pdes/base_pde.py) for the main training loops. We currently only support Adam and [NNCG](https://arxiv.org/abs/2402.01868), but we think there's a lot more to do towards higher-precision optimizers with BWLer!


## Citation
If you find this work useful, please cite it as follows:
```
@inproceedings{
liu2025bwler,
title={{BWL}er: Barycentric Weight Layer Elucidates a Precision-Conditioning Tradeoff for {PINN}s},
author={Jerry Weihong Liu and Yasa Baig and Denise Hui Jean Lee and Rajat Vadiraj Dwaraknath and Atri Rudra and Christopher Re},
booktitle={Workshop on the Theory of AI for Scientific Computing},
year={2025},
url={https://openreview.net/forum?id=rKZkotB0um}
}
```

