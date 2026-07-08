"""Hand-rolled AdamW (Adam with decoupled weight decay, Loshchilov & Hutter
2019). Weight decay is the ingredient that makes grokking happen on these
tasks at all -- without it the network is happy to sit at a memorizing
solution indefinitely (Power et al. 2022).
"""
import numpy as np


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g * g)
            m_hat = self.m[k] / bias1
            v_hat = self.v[k] / bias2
            params[k] -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                     + self.weight_decay * params[k])
