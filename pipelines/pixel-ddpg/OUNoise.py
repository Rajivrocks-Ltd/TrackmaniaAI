import numpy as np


class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


# class OUNoise:
#     def __init__(self,
#                  theta=0.3,
#                  mu=0.0,
#                  sigma=0.9,
#                  dt=1e-2,
#                  x0=None,
#                  size=1,
#                  sigma_min=None,
#                  n_steps_annealing=1000):
#
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.size = size
#         self.num_steps = 0
#
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
#
#         if sigma_min is not None:
#             self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
#             self.c = sigma
#             self.sigma_min = sigma_min
#         else:
#             self.m = 0
#             self.c = sigma
#             self.sigma_min = sigma
#
#     def current_sigma(self):
#         sigma = max(self.sigma_min, self.m * float(self.num_steps) + self.c)
#         return sigma
#
#     def sample(self):
#         x = (
#                 self.x_prev
#                 + self.theta * (self.mu - self.x_prev) * self.dt
#                 + self.current_sigma() * np.sqrt(self.dt) * np.random.normal(size=self.size)
#         )
#         self.x_prev = x
#         self.num_steps += 1
#         return x
#
