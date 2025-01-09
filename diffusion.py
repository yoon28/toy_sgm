import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from data.utils import load_gmm_samples, load_swiss_samples
from model.model import DenoiseModel


class Diffuser:
    def __init__(self, neural_net, device='cpu'):
        self.device = device
        self.diffusion_steps = 40  # Number of steps in the diffusion process
        # Set noising variances betas as in Nichol and Dariwal paper (https://arxiv.org/pdf/2102.09672.pdf)
        self.s = 0.008
        self.timesteps = torch.tensor(range(0, self.diffusion_steps), dtype=torch.float32)
        self.schedule = torch.cos((self.timesteps / self.diffusion_steps + self.s) / (1 + self.s) * torch.pi / 2)**2

        self.baralphas = self.schedule / self.schedule[0]
        self.betas = 1 - self.baralphas / torch.concatenate([self.baralphas[0:1], self.baralphas[0:-1]])
        self.alphas = 1 - self.betas

        self.net = neural_net

    def noise(self, Xbatch, t):
        eps = torch.randn(size=Xbatch.shape)
        noised = (self.baralphas[t] ** 0.5).repeat(1, Xbatch.shape[1]) * Xbatch + ((1 - self.baralphas[t]) ** 0.5).repeat(1, Xbatch.shape[1]) * eps
        return noised, eps

    def sample_ddpm(self, model, nsamples, nfeatures):
        """Sampler following the Denoising Diffusion Probabilistic Models method by Ho et al (Algorithm 2)"""
        with torch.no_grad():
            x = torch.randn(size=(nsamples, nfeatures)).to(self.device)
            xt = [x]
            for t in range(self.diffusion_steps-1, 0, -1):
                predicted_noise = model(x, torch.full([nsamples, 1], t).to(self.device))
                # See DDPM paper between equations 11 and 12
                x = 1 / (self.alphas[t] ** 0.5) * (x - (1 - self.alphas[t]) / ((1 - self.baralphas[t]) ** 0.5) * predicted_noise)
                if t > 1:
                    # See DDPM paper section 3.2.
                    # Choosing the variance through beta_t is optimal for x_0 a normal distribution
                    variance = self.betas[t]
                    std = variance ** (0.5)
                    x += std * torch.randn(size=(nsamples, nfeatures)).to(self.device)
                xt += [x]
            return x, xt

    def sample_ddpm_x0(self, model, nsamples, nfeatures):
        """Sampler that uses the equations in DDPM paper to predict x0, then use that to predict x_{t-1}
        
        This is how DDPM is implemented in HuggingFace Diffusers, to allow working with models that predict
        x0 instead of the noise. It is also how we explain it in the Mixture of Diffusers paper.
        """
        with torch.no_grad():
            x = torch.randn(size=(nsamples, nfeatures)).to(self.device)
            for t in range(self.diffusion_steps-1, 0, -1):
                predicted_noise = model(x, torch.full([nsamples, 1], t).to(self.device))
                # Predict original sample using DDPM Eq. 15
                x0 = (x - (1 - self.baralphas[t]) ** (0.5) * predicted_noise) / self.baralphas[t] ** (0.5)
                # Predict previous sample using DDPM Eq. 7
                c0 = (self.baralphas[t-1] ** (0.5) * self.betas[t]) / (1 - self.baralphas[t])
                ct = self.alphas[t] ** (0.5) * (1 - self.baralphas[t-1]) / (1 - self.baralphas[t])
                x = c0 * x0 + ct * x
                # Add noise
                if t > 1:
                    # Instead of variance = betas[t] the Stable Diffusion implementation uses this expression
                    variance = (1 - self.baralphas[t-1]) / (1 - self.baralphas[t]) * self.betas[t]
                    variance = torch.clamp(variance, min=1e-20)
                    std = variance ** (0.5)
                    x += std * torch.randn(size=(nsamples, nfeatures)).to(self.device)
            return x

def main():
    device = 'cpu'
    n_samples = 3_000
    X0 = load_gmm_samples(n_samples=n_samples)

    # plt.scatter(X0[:, 0], X0[:, 1])
    # plt.show()

    net = DenoiseModel(nfeatures=2, nblocks=4)
    model = Diffuser(neural_net=net)

    nepochs = 300
    batch_size = 300

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=nepochs)

    for epoch in range(nepochs):
        epoch_loss = steps = 0
        for i in range(0, len(X0), batch_size):
            Xbatch = X0[i:i+batch_size]
            timesteps = torch.randint(0, model.diffusion_steps, size=[len(Xbatch), 1])
            noised, eps = model.noise(Xbatch, timesteps)
            predicted_noise = model.net(noised.to(device), timesteps.to(device))
            loss = loss_fn(predicted_noise, eps.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            steps += 1
        print(f"Epoch {epoch} loss = {epoch_loss / steps}")

    print("the end of training")

    Xgen, Xgen_hist = model.sample_ddpm(model.net, n_samples, 2)
    Xgen = Xgen.cpu()
    plt.scatter(X0[:, 0], X0[:, 1], alpha=0.5)
    plt.scatter(Xgen[:, 0], Xgen[:, 1], marker="1", alpha=0.5)
    plt.legend(["Original data", "Generated data"])
    plt.show()

    def draw_frame(i):
        plt.clf()
        Xvis = Xgen_hist[i].cpu()
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker="1", animated=True)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        return fig,
    fig = plt.figure()
    anime = animation.FuncAnimation(fig, draw_frame, frames=40, interval=20, blit=True)
    anime.save('swissroll_generation.mp4', fps=10)

if __name__ == "__main__":
    main()

