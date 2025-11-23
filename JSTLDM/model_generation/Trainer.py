#!/usr/bin/env python
# coding: utf-8
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from Dataset import GeneratedTextureDataset, GeneratedStructureDataset, GeneratedDataset, GeneratedDatasetFixed, \
    GeneratedTextureDatasetFixed, GeneratedStructureDatasetFixed, GeneratedUniformTextureDataset
from hyperparameters import *
from misc import Timer
from Transformations import AddNoiseRandom


def projection(x, f):
    return 0.5 * torch.concatenate([x[:, :1, :, :] - x[:, 1:, :, :] + f, x[:, 1:, :, :] - x[:, :1, :, :] + f], 1)


class Trainer:

    def __init__(self, model, device, pretrained_path=False, experiment_name=None, exp_type="u"):

        self.start_time = time.localtime()
        if experiment_name is None:
            self.result_folder = f"{result_folder}/{time.asctime(self.start_time).replace(' ', '_')}"
        else:
            self.result_folder = f"{result_folder}/{experiment_name}"

        try:
            os.mkdir(self.result_folder)
        except FileExistsError:
            pass
        self.device = device
        self.model = model.to(device)

        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path, map_location=self.device))

        self.loss_data_train = []
        self.loss_data_cartoon = []
        self.loss_data_texture = []

        self.eval_loss = []

        gaussian_noise = AddNoiseRandom(std_range=[0.0, 0.1], percent=0.3)
        # noise_with_random_injection = AddNoiseRandomWithInjection(std_range=[0.0, 0.1], thresholds=(0.3, 0.65))
        # noise_with_random_injection = None
        if exp_type == "u":
            generated_dataset = GeneratedStructureDataset(size=image_size, dataset_size=train_data_size,
                                                          transform=gaussian_noise)
            evaluation_dataset = GeneratedStructureDatasetFixed(size=image_size, dataset_size=eval_data_size,
                                                                transform=None)
        elif exp_type == "v":
            generated_dataset = GeneratedTextureDataset(size=image_size, dataset_size=train_data_size,
                                                        transform=gaussian_noise)
            evaluation_dataset = GeneratedTextureDatasetFixed(size=image_size, dataset_size=eval_data_size,
                                                              transform=None)
        elif exp_type == "v_uniform":
            generated_dataset = GeneratedUniformTextureDataset(size=image_size, dataset_size=train_data_size,
                                                               transform=gaussian_noise)
            evaluation_dataset = GeneratedUniformTextureDataset(size=image_size, dataset_size=eval_data_size,
                                                                transform=None)
        elif exp_type == "u_v":
            generated_dataset = GeneratedDataset(size=image_size, dataset_size=train_data_size,
                                                 transform=gaussian_noise)
            evaluation_dataset = GeneratedDatasetFixed(size=image_size, dataset_size=eval_data_size, transform=None)
        else:
            generated_dataset = GeneratedDataset(size=image_size, dataset_size=train_data_size,
                                                 transform=gaussian_noise)
            evaluation_dataset = GeneratedDatasetFixed(size=image_size, dataset_size=eval_data_size, transform=None)

        self.generated_dataset_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=batch_size_train,
                                                                    shuffle=train_loader_shuffle, pin_memory=True,
                                                                    num_workers=train_loader_workers)

        self.evaluation_dataset_loader = torch.utils.data.DataLoader(evaluation_dataset, batch_size=batch_size_test,
                                                                     shuffle=True, pin_memory=True,
                                                                     num_workers=train_loader_workers)
        # self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction='mean').to(device)

        # misc variables
        self.curr_epoch = 0
        self.train_psnr = 0.0
        self.recovery_psnr = 0.0
        self.eval_psnr = 0.0
        logging.basicConfig(filename=f'{self.result_folder}/progress_info.log', encoding='utf-8', level=logging.INFO)

    def train(self, nb_epochs):
        """
            Trains the model for several epochs
            : param nb_epochs: The number of epochs to train the model
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(20)
        self.model.train()
        timer = Timer()
        for epoch in range(nb_epochs):
            if epoch == 601:
                # restart learning
                self.lr = learning_rate
            self.curr_epoch = epoch
            self._train_with_noise()
            if epoch % 2 == 0:
                self._eval_noiseless()

            if epoch % 5 == 0:
                self._evaluate()

            # update the learning rate
            if (epoch > 0 and epoch % 50 == 0) or (epoch == 25) or (epoch == 630):
                self.lr *= 0.65
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            if epoch > 0 and epoch % 5 == 0:
                self.save_loss()
            if epoch > 0 and epoch % 50 == 0:
                self.save_model(epoch)
            elif epoch % 2 == 0:
                self._save_model()
            logging.info(f"End of epoch = {epoch}, time: {timer}, learning rate = {self.lr}, "
                         f"train_psnr:{self.train_psnr}, recovery_psnr:{self.recovery_psnr}")

    def _train_with_noise(self):
        """
        Train the model for just one epoch
        """
        number_of_batches = len(self.generated_dataset_loader)
        epoch_loss = np.zeros(number_of_batches)
        for batch_idx, (noisy, data) in enumerate(self.generated_dataset_loader):
            noisy = noisy.to(self.device)
            data = data.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # pass the current batch through the model
            output = self.model(noisy)

            # compute the loss and do a gradient descent step
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            epoch_loss[batch_idx] = loss.item()
        mse = epoch_loss.mean()
        self.train_psnr = 10 * np.log10(1.0 / mse)
        self.loss_data_train.append({"epoch": self.curr_epoch, "loss": mse, "psnr": self.train_psnr})

    def _train_with_projection(self):
        """
                Train the model for just one epoch with an added projection
        """
        number_of_batches = len(self.generated_dataset_loader)
        epoch_loss = np.zeros(number_of_batches)
        for batch_idx, (f, data) in enumerate(self.generated_dataset_loader):

            f = f.to(self.device)
            data = data.to(self.device)

            corrupted = self.model(data).detach()
            corrupted = projection(corrupted, f)

            # reset the gradients
            self.optimizer.zero_grad()

            # pass the current batch through the model
            output = self.model(corrupted)

            # compute the loss and do a gradient descent step
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            epoch_loss[batch_idx] = loss.item()
        mse = epoch_loss.mean()
        self.train_psnr = 10 * np.log10(1.0 / mse)
        self.loss_data_train.append({"epoch": self.curr_epoch, "loss": mse, "psnr": self.train_psnr})

    def _eval_noiseless(self):
        """
        Train the model for just one epoch
        """
        number_of_batches = len(self.evaluation_dataset_loader)
        eval_loss = np.zeros(number_of_batches)

        # retrieve the dataset transform
        for batch_idx, (noiseless, data) in enumerate(self.evaluation_dataset_loader):
            noiseless = noiseless.to(self.device)
            data = data.to(self.device)

            # pass the current batch through the model
            output = self.model(noiseless).detach()

            # compute the loss and do a gradient descent step
            loss = self.criterion(output, data)
            eval_loss[batch_idx] = loss.item()
        mse = eval_loss.mean()
        self.recovery_psnr = 10 * np.log10(1.0 / mse)
        # reset dataset transform

    def _evaluate(self):

        number_of_batches = len(self.evaluation_dataset_loader)
        eval_loss = np.zeros(number_of_batches)
        for batch_idx, (noisy, data) in enumerate(self.evaluation_dataset_loader):
            noisy = noisy.to(self.device)
            data = data.to(self.device)

            # pass the current batch through the model
            output = self.model(noisy).detach()

            # compute the loss and do a gradient descent step
            loss = self.criterion(output, data)
            eval_loss[batch_idx] = loss.item()
        mse = eval_loss.mean()
        self.eval_psnr = 10 * np.log10(1.0 / mse)
        self.eval_loss.append({"epoch": self.curr_epoch, "loss": mse, "psnr": self.eval_psnr})

    def save_loss(self):

        # save training loss
        df = pd.DataFrame(self.loss_data_train)
        df.to_csv(f"{self.result_folder}/loss_train.csv")

        # save evaluation loss
        # df = pd.DataFrame(self.eval_loss)
        # df.to_csv(f"{self.result_folder}/loss_eval.csv")

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.result_folder}/model_{epoch}.pth")

    def _save_model(self):
        torch.save(self.model.state_dict(), f"{self.result_folder}/model_LAST.pth")
