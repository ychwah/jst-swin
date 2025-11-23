# -*- coding: utf-8 -*-
"""
@author: Antoine Guennec
"""
import os
import torch
from generate_std_images import generate_std, generate_structure, generate_texture
import torchvision.io


def get_img(img_path="../images/cameraman.tif"):
    img = torchvision.io.read_image(img_path).type(torch.float32) / 255
    return img


class GeneratedDataset(torch.utils.data.Dataset):

    def __init__(self, size=128, dataset_size=4096, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        try:
            structure, texture = generate_std(self.size, self.size)
        except ValueError:
            structure, texture = generate_std(self.size, self.size)
        structure = torch.tensor(structure, dtype=torch.float32)
        structure.unsqueeze_(0)
        texture = torch.tensor(texture, dtype=torch.float32)
        texture.unsqueeze_(0)
        std = torch.concatenate([structure, texture], 0)
        if self.transform is not None:
            corrupted = torch.concatenate([self.transform(structure), self.transform(texture)], 0)
        else:
            corrupted = structure + texture

        return corrupted, std


class GeneratedStructureDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size=4096, size=128, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        structure = generate_structure(self.size, self.size)
        structure = torch.tensor(structure, dtype=torch.float32)
        structure.unsqueeze_(0)
        if self.transform is not None:
            with torch.no_grad():
                corrupted = self.transform(structure)
        else:
            corrupted = structure

        return corrupted, structure


class GeneratedTextureDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size=4096, size=128, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        structure, texture = generate_std(self.size, self.size)
        texture = torch.tensor(texture, dtype=torch.float32)
        texture.unsqueeze_(0)
        if self.transform is not None:
            with torch.no_grad():
                corrupted = self.transform(texture)
        else:
            corrupted = texture

        return corrupted, texture


class GeneratedUniformTextureDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size=4096, size=128, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        texture = generate_texture(self.size, self.size)
        texture = torch.tensor(texture, dtype=torch.float32)
        texture.unsqueeze_(0)
        if self.transform is not None:
            with torch.no_grad():
                corrupted = self.transform(texture)
        else:
            corrupted = texture

        return corrupted, texture


class NaturalImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.root = root if root[-1] == "/" else "".join([root, "/"])
        self.img_list = [img.name for img in os.scandir(self.root + "original") if img.name.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        structure = get_img(f"{self.root}structure/{img_name}")
        texture = get_img(f"{self.root}texture/{img_name}") - 0.5
        x = torch.concatenate([structure, texture], 0)
        if self.transform is not None:
            x = self.transform(x)

        return x


class NaturalStructureDataset(torch.utils.data.Dataset):

    def __init__(self, root, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.root = root if root[-1] == "/" else "".join([root, "/"])
        self.img_list = [img.name for img in os.scandir(self.root + "original") if img.name.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        structure = get_img(f"{self.root}structure/{img_name}")
        if self.transform is not None:
            structure = self.transform(structure)

        return structure


class GeneratedDatasetFixed(torch.utils.data.Dataset):

    def __init__(self, size=128, dataset_size=4096, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

        self.std_library = torch.zeros((self.length, 2, self.size, self.size), dtype=torch.float32)
        self.reset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        std = self.std_library[index]
        if self.transform is not None:
            corrupted = self.transform(std)
        else:
            corrupted = std

        return corrupted, std

    def reset(self):
        for k in range(self.length):
            structure, texture = generate_std(self.size, self.size)
            self.std_library[k, 0, ...] = torch.tensor(structure)
            self.std_library[k, 1, ...] = torch.tensor(texture)


class GeneratedStructureDatasetFixed(torch.utils.data.Dataset):

    def __init__(self, size=128, dataset_size=4096, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

        self.std_library = torch.zeros((self.length, 1, self.size, self.size), dtype=torch.float32)
        self.reset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        structure = self.std_library[index]
        if self.transform is not None:
            corrupted = self.transform(structure)
        else:
            corrupted = structure

        return corrupted, structure

    def reset(self):
        for k in range(self.length):
            structure = generate_structure(self.size, self.size)
            self.std_library[k, 0, ...] = torch.tensor(structure)


class GeneratedTextureDatasetFixed(torch.utils.data.Dataset):

    def __init__(self, size=128, dataset_size=4096, transform=None):
        torch.utils.data.Dataset.__init__(self)
        self.size = size
        self.length = dataset_size
        self.transform = transform

        self.std_library = torch.zeros((self.length, 1, self.size, self.size), dtype=torch.float32)
        self.reset()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        texture = self.std_library[index]
        if self.transform is not None:
            corrupted = self.transform(texture)
        else:
            corrupted = texture

        return corrupted, texture

    def reset(self):
        for k in range(self.length):
            structure, texture = generate_std(self.size, self.size)
            self.std_library[k, 0, ...] = torch.tensor(texture)
