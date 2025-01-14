import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from data.utils import load_gmm_samples, load_swiss_samples
from model.model import DenoiseModel

class ScoreF:
    def __init__(self, neural_net, gmm, device='cpu'):
        self.net = neural_net
        self.device = device
        self.gmm = dict()
        for k, v in gmm.items():
            self.gmm[k] = np.array(gmm[k])
        self.gmm["cov1_inv"] = np.linalg.inv(self.gmm["cov1"])
        self.gmm["cov2_inv"] = np.linalg.inv(self.gmm["cov2"])

    def score_gmm_true1(self, x):
        dgmm1 = - (x - self.gmm["mu1"])[None,] @ self.gmm["cov1_inv"]
        dgmm2 = - (x - self.gmm["mu2"])[None,] @ self.gmm["cov2_inv"]
        dgmm = dgmm1 + dgmm2
        return dgmm

    def score_gmm_true(self, xs, ys):
        xy = np.stack([xs, ys], axis=-1)

        dxy1 = - (xy - self.gmm["mu1"][None, None, ])
        dxy1 = dxy1 @ self.gmm["cov1_inv"]

        dxy2 = - (xy - self.gmm["mu2"][None, None, ])
        dxy2 = dxy2 @ self.gmm["cov2_inv"]

        dxy = dxy1 + dxy2
        return dxy[..., 0], dxy[..., 1]

def main():
    device = 'cuda'
    n_samples = 3_000
    data = {"mu1": [2, 1], "cov1": [[0.5, 0], [0, 0.5]],
            "mu2": [-2, -2], "cov2": [[0.2, 0], [0, 0.2]]}
    X0 = load_gmm_samples(n_samples=n_samples, mu1=data["mu1"], cov1=data["cov1"], mu2=data["mu2"], cov2=data["cov2"])

    plt.figure()
    plt.scatter(X0[:, 0], X0[:, 1])
    # plt.show()

    net = DenoiseModel(nfeatures=2, nblocks=4)
    model = ScoreF(net, data)

    ii, jj = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
    dx, dy = model.score_gmm_true(ii, jj)
    plt.quiver(ii, jj, dx, dy, color='g')

    iip = np.zeros([20, 20])
    jjp = np.zeros([20, 20])
    dxp = np.zeros([20, 20])
    dyp = np.zeros([20, 20])
    for j, y in enumerate(np.arange(-4, 4, 0.4)):
        for i, x in enumerate(np.arange(-4, 4, 0.4)):
            p = np.array([x, y])
            dp = model.score_gmm_true1(p)
            iip[j, i] = x
            jjp[j, i] = y
            dxp[j, i] = dp[0, 0]
            dyp[j, i] = dp[0, 1]
    plt.quiver(iip, jjp, dxp, dyp, color='r')
    plt.show()

    nepochs = 300
    batch_size = 300

    loss_fn = nn.MSELoss()

    for epoch in range(nepochs):
        epoch_loss = steps = 0
        for i in range(0, len(X0), batch_size):
            steps += 1
        print(f"Epoch {epoch} loss = {epoch_loss / steps}")

    print("the end of training")


if __name__ == "__main__":
    main()

