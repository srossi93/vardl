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
from ..logger import BaseLogger


class TrainerRegressor():

    def __init__(self,
                 model: nn.Module,
                 optimizer: str,
                 optimizer_config: Dict,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 device: str,
                 seed: int,
                 logger: BaseLogger=None):

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

        self.logger = logger

        set_seed(seed)

    def compute_nell(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        nell = - n / m * \
            torch.sum(torch.mean(self.model.likelihood.log_cond_prob(Y_true, Y_pred), 0))
        return nell

    def compute_loss(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        return self.compute_nell(Y_pred, Y_true, n, m) + self.model.dkl

    def compute_error(self, Y_pred: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(torch.pow((Y_true - Y_pred), 2))
                          )  # ok for regression

    def train_batch(self, data: torch.Tensor, target: torch.Tensor,
                    train_verbose: bool, train_log_interval: int):

        self.current_iteration += 1

        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)

        loss = self.compute_loss(output, target, len(
            self.train_dataloader.dataset), data.size(0))
        error = self.compute_error(output, target)
        loss.backward()

        if self.current_iteration % train_log_interval == 0 and train_verbose:
            print(colored('Train', 'blue', attrs=['bold']),
                  "|| iter=%5d   loss=%10.0f  dkl=%8.0f  error=%.2f  log_theta_noise_var=%5.2f" %
                  (self.current_iteration, loss.item(), self.model.dkl.item(),
                   error.item(),
                   self.model.likelihood.log_noise_var.item()))

        self.logger.scalar_summary('loss/train', loss, self.current_iteration)
        self.logger.scalar_summary('error/train', error, self.current_iteration)
        self.logger.scalar_summary('model/dkl', self.model.dkl, self.current_iteration)
        self.logger.scalar_summary('model/log_noise_var', self.model.likelihood.log_noise_var,
                                   self.current_iteration)

        self.optimizer.step()

    def train_per_iterations(self, iterations: int,
                             train_verbose: bool, train_log_interval: int):
        """ Implement the logic of training the model. """
        self.model.train()
        dataloader_iterator = iter(self.train_dataloader)
        self.current_epoch += 1

        for i in range(iterations):
            try:
                data, target = next(dataloader_iterator)
            except BaseException:
                del dataloader_iterator
                dataloader_iterator = iter(self.train_dataloader)
                data, target = next(dataloader_iterator)
                self.current_epoch += 1

            self.train_batch(data, target, train_verbose, train_log_interval)

    def train_epochs(self, epochs: int):

        for _ in range(epochs):
            self.current_epoch += 1
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                self.train_batch(data, target)

    def test(self):
        self.model.eval()
        test_nell = 0
        test_error = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss = self.compute_nell(output, target, len(
                    self.test_dataloader.dataset), data.size(0))
                test_nell += batch_loss
                test_error += self.compute_error(output, target)

        test_nell /= len(self.test_dataloader.dataset)
        test_error /= len(self.test_dataloader)

        print(colored('Test', 'green', attrs=['bold']),
              " || iter=%5d   mnll=%10.3f   rmse=%8.3f" % (self.current_iteration, test_nell.item(), test_error.item()))

        self.logger.scalar_summary('loss/test', test_nell, self.current_iteration)
        self.logger.scalar_summary('error/test', test_error, self.current_iteration)

    def fit(self, iterations: int, test_interval: int,
            train_verbose: bool, train_log_interval: int):
        for _ in range(iterations // test_interval):
            self.test()
            self.train_per_iterations(test_interval, train_verbose, train_log_interval)
        self.test()
