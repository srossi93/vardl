# Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..likelihoods import Softmax


class LeNetMNC_MNIST(nn.Module):

    def __init__(self, nmc_test):
        super(LeNetMNC_MNIST, self).__init__()


        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, padding=0, bias=None)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, padding=0, bias=None)
        self.fc1 = nn.Linear(800, 500, bias=None)
        self.fc2 = nn.Linear(500, 10, bias=None)

        self.nmc_train = 1
        self.nmc_test = nmc_test

        self.likelihood = Softmax()
        self.dkl = 0

        self.train(False)

    def save_model(self, path):
        print('INFO - Saving model in %s' % path)
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        print('INFO - Loading model from %s' % path)
        self.load_state_dict(torch.load(path))

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test

    def forward(self, input):
        out = torch.zeros(self.nmc, input.size(0), 10).to(input.device)

        for i in range(self.nmc):
            x = input
            #x = nnf.dropout(input, 0.9, training=True)
            x = nnf.max_pool2d(nnf.relu(self.conv1(x)), (2, 2))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(nnf.relu(self.conv2(x)), (2, 2))
            x = x.view(-1, 800)
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.fc1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = self.fc2(x)

            out[i] = x

        return out

class LeNetMNC_CIFAR10(nn.Module):

    def __init__(self, nmc_test):
        super(LeNetMNC_CIFAR10, self).__init__()


        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, padding=0, bias=None)
        self.conv2 = nn.Conv2d(192, 192, kernel_size=5, padding=0, bias=None)
        self.fc1 = nn.Linear(4800, 1000, bias=None)
        self.fc2 = nn.Linear(1000, 10, bias=None)

        self.nmc_train = 1
        self.nmc_test = nmc_test

        self.likelihood = Softmax()
        self.dkl = 0

        self.train(False)

    def save_model(self, path):
        print('INFO - Saving model in %s' % path)
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        print('INFO - Loading model from %s' % path)
        self.load_state_dict(torch.load(path))

    def train(self, mode=True):
        self.nmc = self.nmc_train if mode else self.nmc_test

    def forward(self, input):
        out = torch.zeros(self.nmc, input.size(0), 10).to(input.device)

        for i in range(self.nmc):
            x = input
            x = nnf.max_pool2d(nnf.relu(self.conv1(x)), (2, 2))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(nnf.relu(self.conv2(x)), (2, 2))
            x = x.view(-1, 4800)
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.fc1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = self.fc2(x)

            out[i] = x

        return out


