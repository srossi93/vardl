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
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from termcolor import colored
from torch.utils.data import DataLoader

from ..layers import BaseBayesianLayer, BayesianConv2d, BayesianLinear
from ..logger import TensorboardLogger
from ..utils import set_seed
from ..utils.exception import VardlRunningTimeException, VardlNaNLossException
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
                 prior_update_interval: int = 0,
                 prior_update_conv2d_type: str = 'layer'):

        #assert optimizer == 'Adam'

        self._logger = logging.getLogger(__name__)
        self.device = device
        self.model = model.to(self.device)

        self._logger.info('Parameters to optimize:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._logger.info('  %s' % name)
        #self._logger.info('Total: %s' % model.trainable_parameters)

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

        self.prior_update_interval = prior_update_interval
        self.is_prior_update = True if prior_update_interval != 0 else False
        self.prior_update_conv2d_type = prior_update_conv2d_type

        # dummy_input = next(iter(test_dataloader))
        # print('Add graph')
        # self.tb_logger.writer.add_graph(self.model, (dummy_input, ), True)

        self.debug = False


    def compute_nell(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        nell = - n / m * \
            torch.sum(torch.mean(self.model.likelihood.log_cond_prob(Y_true, Y_pred), 0))
        return nell

    def compute_kl(self):
        return self.model.dkl * .1/(1 + np.exp(-0.00125 * (self.current_iteration - 30000))) #4500 ok

    def compute_loss(self, Y_pred: torch.Tensor, Y_true: torch.Tensor,
                     n: int, m: int) -> torch.Tensor:
        return self.compute_nell(Y_pred, Y_true, n, m) + self.compute_kl()

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

        if self.debug:
            with torch.autograd.detect_anomaly():
                output = self.model(data)

                loss = self.compute_loss(output, target, len(self.train_dataloader.dataset), data.size(0))
                error = self.compute_error(output, target)
                loss.backward()
        else:
            output = self.model(data)

            loss = self.compute_loss(output, target, len(self.train_dataloader.dataset), data.size(0))
            error = self.compute_error(output, target)
            loss.backward()

        if torch.isnan(loss):
            self._logger.error('At step %d loss became NaN. Run with debug flag to investigate where it was produced'
                               % self.current_iteration)
            raise VardlNaNLossException

        self.optimizer.step()

        # -- Reporting to stdout and to the tb_logger (i.e. tensorboard)
        if self.current_iteration % train_log_interval == 0:
            if train_verbose:
                self._logger.debug('Train: iter=%5d  loss=%01.03e  dkl=%01.03e  error=%.2f ' %
                                   (self.current_iteration, loss.item(), self.compute_kl(), error.item()))

                for name, param in self.model.named_parameters():
                    if param.requires_grad:
            #            self.tb_logger.writer.add_histogram(name,
            #                                             param.clone().cpu().data.numpy(),
            #                                             self.current_iteration, )
                        self.tb_logger.writer.add_histogram(name + '.grad',
                                                         param.grad.clone().cpu().data.numpy(),
                                                         self.current_iteration, )

        self.tb_logger.scalar_summary('loss/train', loss, self.current_iteration)
        self.tb_logger.scalar_summary('loss/train/nll',
                                      self.compute_nell(output,
                                                     target,
                                                     len(self.train_dataloader.dataset),
                                                     data.size(0)), self.current_iteration)
        self.tb_logger.scalar_summary('error/train', error, self.current_iteration)
        self.tb_logger.scalar_summary('model/dkl', self.compute_kl(), self.current_iteration)

        if self.is_prior_update:
            self._prior_update()

    def _prior_update(self):

        if self.current_iteration % self.prior_update_interval == 0:
            #self._logger.debug('Step %s - Updating priors ' % self.current_iteration)
            for child in self.model.modules():
                if isinstance(child, BayesianConv2d):
                    # For conv2d, weights have shape [out_channels, in_channels, kernel_size, kernel_size]
                    prior_means = child.prior_W.mean.view(child.out_channels, child.in_channels,
                                                                child.kernel_size,
                                                                 child.kernel_size)
                    prior_vars = child.prior_W.logvars.view(child.out_channels, child.in_channels,
                                                                       child.kernel_size,
                                                                       child.kernel_size).exp()

                    q_means = child.q_posterior_W.mean.view(child.out_channels, child.in_channels,
                                                                 child.kernel_size,
                                                                 child.kernel_size)
                    q_vars = child.q_posterior_W.logvars.view(child.out_channels, child.in_channels,
                                                                    child.kernel_size,
                                                                    child.kernel_size).exp()

                    new_prior_means = torch.zeros_like(prior_means, device=prior_means.device)
                    new_prior_logvars = torch.zeros_like(prior_vars, device=prior_vars.device)

                    if self.prior_update_conv2d_type == 'layer':
                        m = q_means.mean()
                        s = (q_vars + torch.pow((prior_means - q_means), 2)).mean()
                        new_prior_means.fill_(m)
                        new_prior_logvars.fill_(torch.log(s))

                    if self.prior_update_conv2d_type == 'outchannels':
                        for c_out in range(child.out_channels):
                            m = q_means[c_out].mean()
                            s = (q_vars[c_out] + torch.pow((prior_means[c_out] - q_means[c_out]), 2)).mean()

                            new_prior_means[c_out].fill_(m)
                            new_prior_logvars[c_out].fill_(torch.log(s))

                    if self.prior_update_conv2d_type == 'outchannels+inchannels':
                        for c_out in range(child.out_channels):
                            for c_in in range(child.in_channels):
                                m = q_means[c_out, c_in].mean()
                                s = (q_vars[c_out, c_in] + torch.pow((prior_means[c_out, c_in] - q_means[c_out, c_in]), 2)).mean()

                                new_prior_means[c_out, c_in].fill_(m)
                                new_prior_logvars[c_out, c_in].fill_(torch.log(s))

                    if self.prior_update_conv2d_type == 'outchannels+inchannels+inrows':
                        for c_out in range(child.out_channels):
                            for c_in in range(child.in_channels):
                                for r_in in range(child.in_height):
                                    m = q_means[c_out, c_in, r_in].mean()
                                    s = (q_vars[c_out, c_in, r_in] + torch.pow((prior_means[c_out, c_in, r_in] - q_means[c_out, c_in, r_in]), 2)).mean()

                                    new_prior_means[c_out, c_in, r_in].fill_(m)
                                    new_prior_logvars[c_out, c_in, r_in].fill_(torch.log(s))


                    child.prior_W._mean.data = new_prior_means.view_as(child.prior_W.mean)
                    child.prior_W._logvars.data = new_prior_logvars.view_as(child.prior_W.logvars)


                if isinstance(child, BayesianLinear):
                    new_prior_mean = child.q_posterior_W._mean.data.mean()
                    new_prior_var = (child.q_posterior_W._logvars.data.exp() + torch.pow((child.prior_W._mean.data -
                                                                                          child.q_posterior_W._mean.data),
                                                                                         2)).mean()
                    child.prior_W._mean.data.fill_(new_prior_mean)
                    child.prior_W._logvars.data.fill_(torch.log(new_prior_var))

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
            #print('Test: iter=%5d   mnll=%01.03e   error=%8.3f' %
            #                  (self.current_iteration, test_nell.item(), test_error.item()))
            self._logger.info('Test: iter=%5d   mnll=%01.03e   error=%8.3f' %
                              (self.current_iteration, test_nell.item(), test_error.item()))

        self.tb_logger.scalar_summary('loss/test', test_nell, self.current_iteration)
        self.tb_logger.scalar_summary('error/test', test_error, self.current_iteration)

        return test_nell, test_error

    def fit(self, iterations: int, test_interval: int,
            train_verbose: bool, train_log_interval: int = 1000, time_budget=120):

        best_test_nell, best_test_error = self.test()

        t_start = time.time()
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
                if (time.time() - t_start) / 60 > time_budget:
                    raise VardlRunningTimeException('Interrupting training due to time budget elapsed')

            test_nell, test_error = self.test()
            self.tb_logger.save_model('_final')
            return test_nell, test_error

        except KeyboardInterrupt:
            self._logger.warning('Training interruped by user. Saving current model snapshot')
            self.tb_logger.save_model('_interruped')
            return self.test()

        except VardlRunningTimeException as e:
            self._logger.warning(e)
            return self.test()

    def __adjust_learning_rate(self):
        gamma = self.lr_decay_config['gamma']# 0.0001
        p = self.lr_decay_config['p']#0#0.75

        lr = self.optimizer_config['lr'] * ((1 + gamma * self.current_iteration) ** -p)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.tb_logger.scalar_summary('model/lr', lr, self.current_iteration)

