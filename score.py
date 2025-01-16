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
        self.gmm["det1"] = np.linalg.det(self.gmm["cov1"])
        self.gmm["det2"] = np.linalg.det(self.gmm["cov2"])
        self.gmm["cov1_inv"] = np.linalg.inv(self.gmm["cov1"])
        self.gmm["cov2_inv"] = np.linalg.inv(self.gmm["cov2"])
        self.gmm["norm1"] = 1 / (2 * np.pi * np.sqrt(self.gmm["det1"]))
        self.gmm["norm2"] = 1 / (2 * np.pi * np.sqrt(self.gmm["det2"]))

    def gmm_prob(self, xys):
        if len(xys.shape) == 2:
            xys = xys[..., None] # N 2 1
        diff1 = xys - self.gmm["mu1"][None, :, None] # N 2 1
        exponent1 = -0.5 * diff1.transpose(0, 2, 1) @ self.gmm["cov1_inv"] @ diff1
        prob1 = self.gmm["norm1"] * np.exp(exponent1)

        diff2 = xys - self.gmm["mu2"][None, :, None]
        exponent2 = -0.5 * diff2.transpose(0, 2, 1) @ self.gmm["cov2_inv"] @ diff2
        prob2 = self.gmm["norm2"] * np.exp(exponent2)

        return prob1.squeeze(), prob2.squeeze()

    def score_gmm_true(self, xys, p1, p2):
        if len(xys.shape) == 2:
            xys = xys[..., None] # N 2 1

        dxy1 = xys - self.gmm["mu1"][None, :, None]
        dxy1 = - self.gmm["cov1_inv"] @ dxy1
        xy1 = self.gmm["pi1"] * p1[..., None] * dxy1.squeeze()

        dxy2 = xys - self.gmm["mu2"][None, :, None]
        dxy2 = - self.gmm["cov2_inv"] @ dxy2
        xy2 = self.gmm["pi2"] * p2[..., None] * dxy2.squeeze()

        dxy = xy1.squeeze() + xy2.squeeze()
        prob = (self.gmm["pi1"] * p1 + self.gmm["pi2"] * p2)[..., None]
        score = dxy / prob
        return score[..., 0], score[..., 1]


def main():
    device = 'cuda'
    np.random.seed(4242)
    n_samples = 3_000
    data = {"mu1": [2, 1], "cov1": [[0.5, 0], [0, 0.5]],
            "mu2": [-2, -2], "cov2": [[0.2, 0], [0, 0.2]],
            "pi1": 0.5, "pi2": 0.5}
    X0 = load_gmm_samples(n_samples=n_samples, mu1=data["mu1"], cov1=data["cov1"], mu2=data["mu2"], cov2=data["cov2"])

    plt.figure()
    plt.scatter(X0[:, 0], X0[:, 1])

    net = DenoiseModel(nfeatures=2, nblocks=4, is_score=True).to(device)
    model = ScoreF(net, data)

    xx, yy = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
    xys = np.stack([xx.ravel(), yy.ravel()], axis=1)
    p1, p2 = model.gmm_prob(xys)
    probs = p1 * model.gmm["pi1"] + p2 * model.gmm["pi2"]
    plt.contourf(xx, yy, probs.reshape(xx.shape), levels=50, cmap='viridis', alpha=0.8)

    dx, dy = model.score_gmm_true(xys, p1, p2)
    plt.quiver(xx, yy, dx, dy, color='r')
    plt.show()

    nepochs = 300
    batch_size = 300

    optimizer = optim.Adam(model.net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    loss_history = []

    for epoch in range(nepochs):
        epoch_loss = steps = 0
        for i in range(0, len(X0), batch_size):
            optimizer.zero_grad()
            x = np.random.uniform(-4, 4, size=(batch_size, 2))
            gmm_p1, gmm_p2 = model.gmm_prob(x)
            score_x, score_y = model.score_gmm_true(x, gmm_p1, gmm_p2)
            score_gt = torch.tensor(np.stack([score_x, score_y], axis=1), dtype=torch.float32).to(device)
            tensor = torch.tensor(x, dtype=torch.float32).to(device)
            predictions = model.net(tensor)
            loss = loss_fn(predictions, score_gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch} loss = {epoch_loss / steps}")
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid()
    plt.show()
    print("the end of training")

    # test
    model.net.eval()
    batch_test = 100
    x_test = np.random.uniform(-4, 4, size=(batch_test, 2))
    gmm_p1, gmm_p2 = model.gmm_prob(x_test)
    score_x, score_y = model.score_gmm_true(x_test, gmm_p1, gmm_p2)
    with torch.no_grad():
        tensor_test = torch.tensor(x_test, dtype=torch.float32).to(device)
        predictions = model.net(tensor_test)
    pred = predictions.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.quiver(x_test[:, 0], x_test[:, 1], score_x, score_y, color='blue', alpha=0.6, label="True Scores")
    plt.quiver(x_test[:, 0], x_test[:, 1], pred[:, 0], pred[:, 1], color='red', alpha=0.6, label="Predicted Scores")
    plt.title("True vs Predicted Score Vectors")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    plt.show()

    # langevin dynamics sampling
    batch_test = 3_000
    x_test = np.random.uniform(-4, 4, size=(batch_test, 2))
    x_test = torch.tensor(x_test, dtype=torch.float32).to(device)
    x_history = [x_test.cpu()]
    T = 40
    step_size = 0.1
    with torch.no_grad():
        for _ in range(T):
            z = np.random.randn(batch_test, 2)
            z = torch.tensor(z, dtype=torch.float32).to(device)
            score = model.net(x_test)
            x_test = x_test + step_size * score + np.sqrt(2.0 * step_size) * z
            x_history.append(x_test.cpu())
        x_test = x_test.detach().cpu().numpy()
    plt.scatter(x_test[:, 0], x_test[:, 1], s=5, color='r', alpha=0.5, label="Langevin Samples")
    plt.title("Samples from GMM using Langevin Dynamics")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()
    plt.show()

    def draw_frame(i):
        plt.clf()
        Xvis = x_history[i].cpu()
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker="1", animated=True)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        return fig,
    fig = plt.figure()
    anime = animation.FuncAnimation(fig, draw_frame, frames=T, interval=20, blit=True)
    anime.save('score.mp4', fps=10)

if __name__ == "__main__":
    main()

