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

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from termcolor import colored
from torch.utils.data import DataLoader

from ..layers import BaseBayesianLayer
from ..logger import TensorboardLogger
from ..utils import set_seed
import logging

class TrainerClassifier():

    def __init__(self,
                 model: nn.Module,
                 optimizer: str,
                 optimizer_config: Dict,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 device: str,
                 seed: int,
                 logger: TensorboardLogger=None,
                 lr_decay_config: Dict = None,
                 prior_optimization: int = 10000):

        #assert optimizer == 'Adam'

        self._logger = logging.getLogger(__name__)
        self.device = device
        self.model = model.to(self.device)

        self._logger.info('Parameters to optimize:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._logger.info('  %s' % name)

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                        **optimizer_config)
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        **optimizer_config)

        self.current_epoch = 0
        self.current_iteration = 0

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.tb_logger = logger
        self.optimizer_config = optimizer_config

        set_seed(seed)
        self.lr_decay_config = lr_decay_config

        self.prior_optimization = prior_optimization

        # dummy_input = next(iter(test_dataloader))
        # print('Add graph')
        # self.logger.writer.add_graph(self.model, (dummy_input, ), True)


    def compute_nell(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        nell = - n / m * \
            torch.sum(torch.mean(self.model.likelihood.log_cond_prob(Y_true, Y_pred), 0))
        return nell

    def compute_loss(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        return self.compute_nell(Y_pred, Y_true, n, m) + (self.model.dkl)

    def compute_error(self, Y_pred: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:

        likel_pred = self.model.likelihood.predict(Y_pred)
        mean_prediction = torch.mean(likel_pred, 0)
        prediction = torch.argmax(mean_prediction, 1)
        target = torch.argmax(Y_true, 1)
        correct = torch.sum(prediction.data == target.data)

        #print(correct.cpu(), Y_true.cpu().size(0))

        return 1. - correct.float() / (Y_true.size(0))


    def train_batch(self, data: torch.Tensor, target: torch.Tensor,
                    train_verbose: bool, train_log_interval: int):

        self.current_iteration += 1

        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)

        loss = self.compute_loss(output, target, len(self.train_dataloader.dataset), data.size(0))
        error = self.compute_error(output, target)
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100000)

        self.optimizer.step()

        # -- Reporting to stdout and to the logger (i.e. tensorboard)
        if self.current_iteration % train_log_interval == 0:
            if train_verbose:
                self._logger.debug('Train: iter=%5d  loss=%01.03e  dkl=%01.03e  error=%.2f ' %
                                   (self.current_iteration, loss.item(), self.model.dkl.item(), error.item()))

            #for name, param in self.model.named_parameters():
            #    if param.requires_grad:
            #        self.logger.writer.add_histogram(name,
            #                                         param.clone().cpu().data.numpy(),
            #                                         self.current_iteration, )
            #        self.logger.writer.add_histogram(name + '.grad',
            #                                         param.grad.clone().cpu().data.numpy(),
            #                                         self.current_iteration, )

        self.tb_logger.scalar_summary('loss/train', loss, self.current_iteration)
        self.tb_logger.scalar_summary('loss/train/nll',
                                      self.compute_nell(output,
                                                     target,
                                                     len(self.train_dataloader.dataset),
                                                     data.size(0)), self.current_iteration)
        self.tb_logger.scalar_summary('error/train', error, self.current_iteration)
        self.tb_logger.scalar_summary('model/dkl', self.model.dkl, self.current_iteration)

#        if self.current_iteration % self.prior_optimization:
#            #print('INFO - Updating priors')
#            for child in self.model.modules():
#                if issubclass(type(child), BaseBayesianLayer):
#                    new_prior_mean = child.q_posterior_W._mean.data.mean()
#                    new_prior_var = (child.q_posterior_W._logvars.data.exp() + torch.pow((child.prior_W._mean.data -
#                                                                                   child.q_posterior_W._mean.data),
#                                                                                         2)).mean()
#                    child.prior_W._mean.data.fill_(new_prior_mean)
#                    child.prior_W._logvars.data.fill_(torch.log(new_prior_var))

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
                #del dataloader_iterator
                dataloader_iterator = iter(self.train_dataloader)
                data, target = next(dataloader_iterator)
                self.current_epoch += 1

            self.train_batch(data, target, train_verbose, train_log_interval)

    def train_epochs(self, epochs: int):

        for _ in range(epochs):
            self.current_epoch += 1
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                self.train_batch(data, target)

    def test(self, verbose=True):
        self.model.eval()
        test_nell = 0
        test_error = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                batch_loss = self.compute_nell(output, target, 1, 1)
                                               #len(self.test_dataloader.dataset), data.size(0))
                test_nell += batch_loss
                #test_correct += self.compute_error(output, target)#
                test_error += self.compute_error(output, target)#

        test_nell /= len(self.test_dataloader.dataset)
        test_error = test_error/len(self.test_dataloader)

        #test_error /= len(self.test_dataloader)
        if verbose:
            self._logger.info('Test: iter=%5d   mnll=%01.03e   error=%8.3f' %
                              (self.current_iteration, test_nell.item(), test_error.item()))

        self.tb_logger.scalar_summary('loss/test', test_nell, self.current_iteration)
        self.tb_logger.scalar_summary('error/test', test_error, self.current_iteration)

        return test_nell, test_error

    def fit(self, iterations: int, test_interval: int,
            train_verbose: bool, train_log_interval: int = 1000):

        best_test_nell, best_test_error = self.test()

        try:
            for _ in range(iterations // test_interval):

                self.train_per_iterations(test_interval, train_verbose, train_log_interval)

                test_nell, test_error = self.test()
                if test_nell < best_test_nell and test_error < best_test_error:
                    self._logger.info('Current snapshot (MNLL: %.3f - ERR: %.3f) better than previous.' %
                                      (test_nell, test_error))
                    self.tb_logger.save_model('_best')
                    best_test_error = test_error
                    best_test_nell = test_nell

                # Adjust learning rate
                self.__adjust_learning_rate()

            test_nell, test_error = self.test()
            self.tb_logger.save_model('_final')
            return test_nell, test_error

        except KeyboardInterrupt:
            self._logger.warning('Training interruped by user. Saving current model snapshot')
            self.tb_logger.save_model('_interruped')


    def __adjust_learning_rate(self):
        gamma = self.lr_decay_config['gamma']# 0.0001
        p = self.lr_decay_config['p']#0#0.75

        lr = self.optimizer_config['lr'] * ((1 + gamma * self.current_iteration) ** -p)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.tb_logger.scalar_summary('model/lr', lr, self.current_iteration)

