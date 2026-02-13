import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule


class ATCSkillModel(PyroModule):
    def __init__(self, n_features, n_drugs, rank=32, prior_sigma=0.1):
        super().__init__()
        self.n_features = n_features
        self.n_drugs = n_drugs
        self.rank = rank
        self.prior_sigma = prior_sigma

    def model(self, x, y=None):
        sigma = self.prior_sigma

        U = pyro.sample("U", dist.Normal(0., sigma)
                        .expand([self.n_features, self.rank]).to_event(2))
        V = pyro.sample("V", dist.Normal(0., sigma)
                        .expand([self.n_drugs, self.rank]).to_event(2))
        W = pyro.sample("W", dist.Normal(0., sigma)
                        .expand([self.n_features, self.n_drugs]).to_event(2))
        
        b = pyro.sample(
            "b",
            dist.Normal(x.new_zeros(self.n_drugs), x.new_ones(self.n_drugs) * 0.1).to_event(1)
        )

        with pyro.plate("data", x.shape[0]):
            base = torch.matmul(x, W)                      # [B, M]
            latent = torch.matmul(torch.matmul(x, U), V.T)            # [B, M]
            logits = base + latent + b
            pyro.sample("obs", dist.Bernoulli(logits=logits).to_event(1), obs=y)

    def guide(self, x, y=None):
        # W variational params
        mW = pyro.param("mW", 0.01 * torch.randn(self.n_features, self.n_drugs))
        sW = pyro.param("sW", 0.1 * torch.ones(self.n_features, self.n_drugs),
                        constraint=dist.constraints.positive)
        pyro.sample("W", dist.Normal(mW, sW).to_event(2))

        # U variational params
        mU = pyro.param("mU", 0.01 * torch.randn(self.n_features, self.rank))
        sU = pyro.param("sU", 0.1 * torch.ones(self.n_features, self.rank),
                        constraint=dist.constraints.positive)
        pyro.sample("U", dist.Normal(mU, sU).to_event(2))

        # V variational params
        mV = pyro.param("mV", 0.01 * torch.randn(self.n_drugs, self.rank))
        sV = pyro.param("sV", 0.1 * torch.ones(self.n_drugs, self.rank),
                        constraint=dist.constraints.positive)
        pyro.sample("V", dist.Normal(mV, sV).to_event(2))
    
        mb = pyro.param("mb", torch.zeros(self.n_drugs, device=x.device))
        sb = pyro.param("sb", 0.1 * torch.ones(self.n_drugs, device=x.device),
                        constraint=dist.constraints.positive)
        pyro.sample("b", dist.Normal(mb, sb).to_event(1))