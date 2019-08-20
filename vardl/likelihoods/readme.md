# Likelihoods

This module provides basic classes for likelihoods. User-defined likelihood need to derive from the abstract class 
`BaseLikelihood` and implement two abstract class methods (`BaseLikelihood.log_cond_prob` and `BaseLikelihood.predict`).

An example of available likelihood is the `Softmax`, defined as:
```python
class Softmax(BaseLikelihood):

    def __init__(self):
        super(Softmax, self).__init__()

    def log_cond_prob(self, output: torch.Tensor,
                      latent_val: torch.Tensor) -> torch.Tensor:
        return torch.sum(output * latent_val, 2) - torch.logsumexp(latent_val, 2)

    def predict(self, latent_val):
        logprob = latent_val - torch.unsqueeze(torch.logsumexp(latent_val, 2), 2)
        return logprob.exp()
```

Currently available likelihood are:
- `vardl.likelihoods.Gaussian`
- `vardl.likelihoods.Softmax`