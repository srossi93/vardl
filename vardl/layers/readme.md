# Layers

This module implements different types of variational layers. 
They all follow the Pytorch naming convention (`Linear`, `Conv2d`, etc.).
All layers should derive from `BaseVariationalLayer` and they need to implement the forward function and the 
computation of the KL divergence within the layer.
For additionl flexibility, all layers should also create the prior and the posterior distributions (from `.
.distributions`) inside the `__init__()` function based on the available distribution (see `..distribution
.available_distributions`)

## Current available layers
Current available layers are
- `vardl.layers.VariationalLinear`
- `vardl.layers.VariationalConv2d`

## WIP and TODOs:
- Variational fastfood