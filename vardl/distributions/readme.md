# Distribution

This module implements a set of differnet probability distribution that can be used as either priors or variational 
posteriors.
All distributions need to derive from the abstract class `BaseDistribution` and need to implement three abstract 
method (`sample(...)`, `sample_local_reparam_linear(...)` and `sample_local_reparam_conv2d(...)`). 
If one of this sampling methods is not possible, the user should be notifyied and either fall back to simpler 
strategies or fail loudly.
For run-time awareness, each derived distribution should set the attributes `has_local_reparam_linear` and 
`has_local_reparam_conv2d` to `True` or `False` accordingly. 

## Matrix Gaussian Distribution
A matrix Gaussian distribution can have multiple approximations. For this reason `MultivariateGaussianDistribution` is 
implemeted as (again) abstract class and all its different approximation should derive from it. 

## KL Divergence
User should also define the KL divergence for all possible combinations of distribution. 
Once defined the KL divergence needs to be registered using the decorator `@register_kl
(q, p)`.
See documentation of PyTorch 1.0 for additional information


## Current available distributions
Current available distributions are:
- `vardl.distributions.FullyFactorizedMatrixGaussian` 
- `vardl.distributions.FullCovariancedMatrixGaussianDistribution` 
- `vardl.distributions.LowRankCovarianceMatrixGaussian`
