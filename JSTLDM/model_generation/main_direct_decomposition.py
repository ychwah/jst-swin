#!/usr/bin/env python
# coding: utf-8
import torch
import time
import os
import pandas as pd
from hyperparameters import *
from Dataset import GeneratedDataset, GeneratedDatasetFixed
import torch.optim as optim
import logging
import numpy as np
from misc import Timer
from GS_DRUNet import UNetRes

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_num_threads(20)


class TrainerDirectDecomposition:

    def __init__(self, model, device, pretrained_path=None, experiment_name=None):

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

        generated_dataset = GeneratedDataset(size=image_size, dataset_size=train_data_size, transform=None)

        self.generated_dataset_loader = torch.utils.data.DataLoader(generated_dataset, batch_size=batch_size_train,
                                                                    shuffle=train_loader_shuffle, pin_memory=True,
                                                                    num_workers=train_loader_workers)
        # self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss(reduction='mean').to(device)

        # misc variables
        self.curr_epoch = 0
        self.train_psnr = 0.0
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
            self.curr_epoch = epoch
            self._train_direct()

            # update the learning rate
            if (epoch > 0 and epoch % 50 == 0) or (epoch == 25):
                self.lr *= 0.65
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

            if epoch > 0 and epoch % 5 == 0:
                self.save_loss()
            if epoch > 0 and epoch % 50 == 0:
                self.save_model(epoch)
            elif epoch % 2 == 0:
                self._save_model()
            logging.info(
                f"End of epoch = {epoch}, time: {timer}, learning rate = {self.lr}, train_psnr:{self.train_psnr}")

    def _train_direct(self):
        """
        Train the model for just one epoch
        """
        number_of_batches = len(self.generated_dataset_loader)
        epoch_loss = np.zeros(number_of_batches)
        for batch_idx, (input_image, data) in enumerate(self.generated_dataset_loader):
            input_image = input_image.to(self.device)
            data = data.to(self.device)

            # reset the gradients
            self.optimizer.zero_grad()

            # pass the current batch through the model
            output = self.model(input_image)

            # compute the loss and do a gradient descent step
            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()
            epoch_loss[batch_idx] = loss.item()
        mse = epoch_loss.mean()
        self.train_psnr = 10 * np.log10(1.0 / mse)
        self.loss_data_train.append({"epoch": self.curr_epoch, "loss": mse, "psnr": self.train_psnr})

    def save_loss(self):
        # save training loss
        df = pd.DataFrame(self.loss_data_train)
        df.to_csv(f"{self.result_folder}/loss_train.csv")

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), f"{self.result_folder}/model_{epoch}.pth")

    def _save_model(self):
        torch.save(self.model.state_dict(), f"{self.result_folder}/model_LAST.pth")


if __name__ == "__main__":

    if torch.cuda.is_available():
        device_ = torch.device("cuda:0")
        print("Running on the GPU : {}".format(torch.cuda.get_device_name(device_)))
    else:
        device_ = torch.device("cpu")
        print("Running on the CPU")

    direct_model = UNetRes(in_nc=1, out_nc=2, nc=[64, 128, 256, 512], nb=3, act_mode='E', downsample_mode='strideconv',
                           upsample_mode='convtranspose')

    trainer = TrainerDirectDecomposition(direct_model, device_, pretrained_path=f"{result_folder}/Direct_Decomposition/model_600.pth", experiment_name="Direct_Decomposition")
    trainer.train(n_epochs)