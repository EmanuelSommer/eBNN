# Towards E-Value Based Stopping Rules for Bayesian Deep Ensembles

This repository contains the code for the paper ***Towards E-Value Based Stopping Rules for Bayesian Deep Ensembles*** presented as oral presentation at the OPTIMAL Workshop at
AISTATS 2026.

### Setup: 

1. The code of the paper [*Can Microcanonical Langevin Dynamics Leverage Mini-Batch Gradient Noise?* by Sommer et al. 2026](https://arxiv.org/abs/2602.06500) will be used to generate the MCMC samples with the pSMILE and MILE samplers. Install the code and use the provided configs in `configs/` to get the samples and predictions (note that some of the configs require a powerful GPU setup and may take considerable time to run).
2. In addtion to the above codebase & its dependencies only `plotnine>=0.15.1` is required.
3. The results for this paper can then be reproduced with the `main_class.py` and `main_regr.py` scripts in the root directory (interactive use via `# %%` is recommended).


### Experiment overview:

- ViT (22M) on Imagenette, sampler: pSMILE
- Resnet7 on CIFAR-10, sampler: pSMILE
- MLP on UCI datasets, sampler: MILE
    - Bikesharing (distributional regression)
    - Income (binary classification)