# Guided by Evidence: E-Values for the Construction and Analysis of Minimal Bayesian Deep Ensembles

### Setup: 

1. The [code](https://anonymous.4open.science/r/pSMILE/README.md) of this [paper](https://arxiv.org/abs/2602.06500) will be used to generate the MCMC samples with the pSMILE and MILE samplers. Install the code and use the provided configs in `configs/` to get the samples and predictions.
2. In addtion to the above codebase & its dependencies only `plotnine>=0.15.1` is required.
3. The results for this paper can then be reproduced with the `main_class.py` and `main_regr.py` scripts in the root directory.


### Experiment overview:

- ViT (22M) on Imagenette, sampler: pSMILE
- Resnet7 on CIFAR-10, sampler: pSMILE
- MLP on UCI datasets, sampler: MILE
    - Bikesharing (distributional regression)
    - Income (binary classification)