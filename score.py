import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from data.utils import load_gmm_samples, load_swiss_samples
from model.model import DenoiseModel

class Score:
    def __init__(self, neural_net, gmm, device='cpu'):
        self.net = neural_net
        self.device = device
        self.gmm = gmm

    def score_gmm(self):
        # TO DO
        pass

def main():
    device = 'cpu'
    n_samples = 3_000
    data = {"mu1": [2, 2], "cov1": [[0.5, 0], [0, 0.5]],
            "mu2": [-2, -2], "cov2": [[0.2, 0], [0, 0.2]]}
    X0 = load_gmm_samples(n_samples=n_samples, mu1=data["mu1"], cov1=data["cov1"], mu2=data["mu2"], cov2=data["cov2"])

    # plt.scatter(X0[:, 0], X0[:, 1])
    # plt.show()

    net = DenoiseModel(nfeatures=2, nblocks=4)
    model = Score(net, data)

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

