from sklearn.datasets import make_swiss_roll
import torch
import numpy as np

def load_swiss_samples(n_samples=3_000, random_seed=None):
    x, _ = make_swiss_roll(n_samples=n_samples, noise=0.5, random_state=random_seed)

    # Make two-dimensional to easen visualization
    x = x[:, [0, 2]]
    x = (x - x.mean()) / (x.std() * 0.5)
    X = torch.tensor(x, dtype=torch.float32)
    return x

def load_gmm_samples(n_samples=3_000, 
                    mu1=[2, 2], cov1=[[0.5, 0], [0, 0.5]],
                    mu2=[-2, -2], cov2=[[0.2, 0], [0, 0.2]]):
    assert n_samples % 2 == 0
    x1 = np.random.multivariate_normal(mean=mu1, cov=cov1, size=n_samples//2)
    x2 = np.random.multivariate_normal(mean=mu2, cov=cov2, size=n_samples//2)
    x = np.concatenate([x1, x2], axis=0)
    np.random.shuffle(x)
    return torch.tensor(x, dtype=torch.float32)