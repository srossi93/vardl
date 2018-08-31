r"""
   Copyright 2018 Simone Rossi, Maurizio Filippone

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from termcolor import colored

from ..utils import set_seed


class TrainerRegressor():

    def __init__(self,
                 model: nn.Module,
                 optimizer: str,
                 optimizer_config: Dict,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 device: str,
                 seed: int):
        assert device == 'cuda' or device == 'cpu'
        assert optimizer == 'Adam'

        self.device = device
        self.model = model.to(self.device)

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        **optimizer_config)
        self.current_epoch = 0
        self.current_iteration = 0

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        set_seed(seed)

    def compute_nell(self, Y_pred: torch.Tensor, Y_true: torch.Tensor, n: int, m: int):
        nell = - n / m * torch.sum(torch.mean(self.model.likelihood.log_cond_prob(Y_true, Y_pred), 0))
        return nell

    def compute_loss(self, Y_pred: torch.Tensor, Y_true: torch.Tensor, n: int, m: int):
        return self.compute_nell(Y_pred, Y_true, n, m) + self.model.dkl

    def compute_error(self, Y_pred, Y_true):
        return torch.sqrt(torch.mean(torch.pow((Y_true - Y_pred), 2)))  # ok for regression



    def train_batch(self, data, target):
        """ Implement the logic of training with one batch. """
        self.current_iteration += 1

        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.compute_loss(output, target, len(self.train_dataloader.dataset), data.size(0))
        loss.backward()

        train_log_interval = 100
        if self.current_iteration % train_log_interval == 0:
            print(colored('Train', 'blue', attrs=['bold']),
                  "|| iter=%5d   loss=%10.0f  error=%.2f  log_theta_noise_var=%5.2f" %
                  (self.current_iteration, loss.item(), self.compute_error(output, target),  self.model.likelihood.log_noise_var.item()))
        self.optimizer.step()

        #print(loss)
        #print('Batch done')

    def train(self):
        """ Implement the logic of training the model. """
        self.model.train()
        for _ in range(self.config.epoch):
            self.cur_epoch += 1
            self.train_epoch()
            self.test()

    def train_epoch(self):
        """ Implement the logic of training one epoch. """
        for batch_idx, (data, target) in tqdm(
                enumerate(self.data_loader.train_loader)):
            self.train_batch(batch_idx, data, target)



    def test(self):
        """ Implement the logic of evaluating the test set performance. """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                batch_loss, batch_correct = self.test_batch(data, target)
                test_loss += batch_loss
                correct += batch_correct

        test_loss /= len(self.data_loader.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
              format(
                  test_loss, correct, len(
                      self.data_loader.test_loader.dataset),
                  100. * correct / len(self.data_loader.test_loader.dataset)))

    def test_batch(self, data, target):
        """ Implement the logic of testing with one batch. """
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        batch_loss = F.nll_loss(
            output, target, size_average=False).item()  # sum up batch loss
        pred = output.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        batch_correct = pred.eq(target.view_as(pred)).sum().item()

        return batch_loss, batch_correct