import math
import numpy as np
import torch as t
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions import MultivariateNormal as MVN

#import sys; sys.path.append("..")
from tpp_trace import *
import utils as u
import tensor_ops as tpp


Norm = lambda mu, var : WrappedDist(Normal, mu, var)

class TVI(nn.Module) :
    """
    Tensorised variational inference. Takes a generator (P), approx posterior (Q), 
    and data (x) as input and returns the ELBO over all k^n sample combinations.
    
    arg:
        - **p:** the generative model, function calling a `trace` object.
        - **q:** the approx posterior, an `nn.module` with a `sample` trace function
        - **k:** the number of simultaneous samples / copies of each latent
        - **x:** data from the output nodes in `p`
        - **nProtectedDims:** number of placeholder dimensions that traces create
    """
    def __init__(self, p, q, k, x, nProtectedDims):
        super().__init__()
        self.p = p
        self.q = q
        self.k = k
        self.nProtected = nProtectedDims
        
        self.data_dict = {}
        self.data_dict["__c"] = []
        self.data = nn.Parameter(x, requires_grad=False) 
        

    def forward(self):
        """
            1. s = sample Q
            2. lp_Q = eval Q.logprob(s)
            3. lp_P = eval P.logprob(s)
            4. F = lp_P - lp_Q
            5. loss = combine Fs
        """
        self.data_dict["__c"].append(self.data)
        # init traces at each step
        sample_trace = sampler(self.k, self.nProtected, data={"__c": self.data})
        # sample recognition model Q -> Q-sample and Q-logprobs
        self.q.sample(sample_trace)
        # Pass Q samples to new trace
        eval_trace = evaluator(sample_trace, self.nProtected, data={"__c": self.data})
        # compute P logprobs 
        self.p(eval_trace)
        
        sum_out_pos(eval_trace)
        sum_out_pos(sample_trace)
        # align dims in Q
        sample_trace.trace.out_dicts = rename_placeholders(eval_trace, sample_trace)
        
        # to ratio land: P.log_probs - Q.log_probs (just the latents)
        tensors = subtract_latent_log_probs(eval_trace, sample_trace)
        
        # combine gives loss
        loss_dict = tpp.combine_tensors(tensors)
        assert(len(loss_dict.keys()) == 1)
        key = next(iter(loss_dict))

        return loss_dict[key]


def setup_and_run(tvi, ep=2000, eta=1) :
    optimiser = t.optim.Adam(tvi.q.parameters(), lr=eta) # optimising q only    
    optimise(tvi, optimiser, ep)
    
    return tvi


def optimise(tvi, optimiser, eps) :
    for i in range(eps):
        optimiser.zero_grad()
        loss = - tvi() 
        loss.backward()
        optimiser.step()
        

def sample_generator(nProtected, P, dataName="__c") :
    """
    Our index-aware tensor product. Takes indexed log_prob tensors as input,
    sums out all indices, and returns dict with one scalar, the loss.
    
    arg:
        - **nProtected (int):** number of positional dimensions for trace to maintain
        - **P (function):** the generative model, defined as a sequence of calls to a trace
    optional kwargs:
        - **dataName (str):** name of the data dimension in P
    """
    k = 1
    trp = sampler(k, nProtected, data={})
    P(trp)
    return trp.trace.out_dicts["sample"][dataName] \
            .squeeze(0)


"""
    Example with factorised chain model. No plates.
    a ~ N([1],[3])
    b ~ N(a,[3])
    c ~ N(b,[3])
"""
if __name__ == "__main__" :
    n = 3
    scale = n
    k = 2
    nProtected = 2
    epochs = 5000 
    lr = 0.1
    TRUE_MEAN_A = 10
    
    # Prior
    # a -> b -> c observed
    def chain_dist(trace, n=3):
        a = trace["a"](Norm(t.ones(n) * TRUE_MEAN_A, scale))
        b = trace["b"](Norm(a, scale))
        c = trace["c"](Norm(b, scale))

        return c
    
    # Q: recognition model. 
    # a placeholder module for params
    class ChainQ(nn.Module):
        def __init__(self, n=3):
            super().__init__()
            self.mean_a = nn.Parameter(t.ones(n))
            self.mean_b = nn.Parameter(t.ones(n))
            self.logscale_a = nn.Parameter(t.ones(n)) 
            self.logscale_b = nn.Parameter(t.ones(n))

        def sample(self, trace) :
            trace["a"](Norm(self.mean_a, t.exp(self.logscale_a)))
            trace["b"](Norm(self.mean_b, t.exp(self.logscale_b)))
        
    def get_error_on_a(a_mean, n, tvi) :
        a_mean = t.ones(n) * a_mean
        return a_mean - tvi.q.mean_a
           
    Q = ChainQ()
    P = chain_dist
    
    x = sample_generator(nProtected, P, dataName="__c")
    tvi = setup_and_run(TVI(P, Q, k, x, nProtected), epochs, eta=lr)
    error = get_error_on_a(TRUE_MEAN_A, n, tvi)
    
    mean_error = error.abs().sum() / 3 
    error_percent = mean_error / TRUE_MEAN_A * 100
    print(f"Error on the estimate of A's mean: {error_percent}%")