import torchvision.transforms as trf
import torch
import numbers
from collections.abc import Sequence


def _setup_size(size, error_msg="Plop"):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, input_image: torch.tensor):

        if len(input_image.shape) == 2:
            n, m = input_image.shape
        else:
            _, n, m = input_image.shape

        start_x = torch.randint(0, n - self.size + 1, size=(1,)).item()
        start_y = torch.randint(0, m - self.size + 1, size=(1,)).item()
        out = input_image[..., start_x: start_x + self.size, start_y: start_y + self.size]

        return out


class AddNoise(torch.nn.Module):

    def __init__(self, std_range=(0, 1.0)):
        super().__init__()
        self.std_range = std_range

    def forward(self, x: torch.tensor):
        if len(x.shape) == 3 and x.shape[0] == 2:
            sigma_1 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
            sigma_2 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
            noise = torch.concatenate(
                [torch.normal(mean=torch.zeros((1,) + x.shape[1:]), std=sigma) for sigma in [sigma_1, sigma_2]], 0)
            out = x + noise
        else:
            sigma = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
            noise = torch.normal(mean=torch.zeros_like(x), std=sigma)
            out = x + noise.to(x.device)
        return out


class AddNoiseRandom(torch.nn.Module):

    def __init__(self, std_range=(0, 1.0), percent=0.3):
        super().__init__()
        self.std_range = std_range
        self.percent = percent

    def forward(self, x: torch.tensor):
        coin_flip = torch.FloatTensor(1).uniform_(0, 1).item()
        if coin_flip >= self.percent:
            if len(x.shape) == 3 and x.shape[0] == 2:
                sigma_1 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                sigma_2 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                noise = torch.concatenate(
                    [torch.normal(mean=torch.zeros((1,) + x.shape[1:]), std=sigma) for sigma in [sigma_1, sigma_2]], 0)
                out = x + noise
            else:
                sigma = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                noise = torch.normal(mean=torch.zeros_like(x), std=sigma)
                out = x + noise.to(x.device)
        else:
            out = x
        return out


class AddNoiseRandomWithInjection(torch.nn.Module):

    def __init__(self, std_range=(0, 1.0), thresholds=(0.3, 0.65)):
        super().__init__()
        self.std_range = std_range
        self.t0 = thresholds[0]
        self.t1 = thresholds[1]

    def forward(self, x: torch.tensor):
        coin_flip = torch.FloatTensor(1).uniform_(0, 1).item()
        if coin_flip >= self.t1:
            if len(x.shape) == 3 and x.shape[0] == 2:
                sigma_1 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                sigma_2 = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                noise = torch.concatenate(
                    [torch.normal(mean=torch.zeros((1,) + x.shape[1:]), std=sigma) for sigma in [sigma_1, sigma_2]], 0)
                out = x + noise
            else:
                sigma = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                noise = torch.normal(mean=torch.zeros_like(x), std=sigma)
                out = x + noise.to(x.device)
        elif self.t0 <= coin_flip < self.t1:
            # structure / texture injection
            if len(x.shape) == 3 and x.shape[0] == 2:
                lamnda_1 = torch.FloatTensor(1).uniform_(-0.25, 0.25).item()
                lambda_2 = torch.FloatTensor(1).uniform_(-0.25, 0.25).item()
                noise = torch.concatenate(
                    [lamnda_1 * x[1:, ...], lambda_2 * x[:1, ...]], 0)
                out = x + noise
            # if
            else:
                sigma = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
                noise = torch.normal(mean=torch.zeros_like(x), std=sigma)
                out = x + noise.to(x.device)
        else:
            out = x
        return out


class AddPoissonNoise(torch.nn.Module):

    def __init__(self, std_range=(0, 0.15)):
        super().__init__()
        self.std_range = std_range

    def forward(self, input_image: torch.tensor):
        level = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.rand(size=input_image.shape)
        output = input_image.detach().clone()
        output[noise < level] = 0.0
        output[noise > (1 - level)] = 1.0
        return output


class AddSpeckleNoise(torch.nn.Module):

    def __init__(self, std_range=(0, 0.15)):
        super().__init__()
        self.std_range = std_range

    def forward(self, input_image: torch.tensor):
        noise_level = torch.FloatTensor(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise1 = torch.rand(size=input_image.shape)
        noise2 = torch.empty(size=input_image.shape, dtype=torch.float32).uniform_(0.5, 1.5)
        noise2[noise1 > noise_level] = 1.0
        output = input_image * noise2
        output[output < 0.0] = 0.0
        output[output > 1.0] = 1.0
        return output
