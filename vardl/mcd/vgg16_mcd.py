#  Copyright (C) 2018   Simone Rossi <simone.rossi@eurecom.fr>
# 	  	       Maurizio Filippone <maurizio.filippone@eurecom.fr>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch
import torch.nn as nn
import torch.nn.functional as nnf

from ..likelihoods import Softmax

class VGG16MCD_CIFAR10(nn.Module):

    def __init__(self, nmc_test):
        super(VGG16MCD_CIFAR10, self).__init__()

        self.conv1_1 = nn.Conv2d(3,   32,  kernel_size=3, stride=1, padding=1,)
        self.conv1_2 = nn.Conv2d(32,  32,  kernel_size=3, stride=1, padding=1,)
        self.conv2_1 = nn.Conv2d(32,  64,  kernel_size=3, stride=1, padding=1,)
        self.conv2_2 = nn.Conv2d(64,  64,  kernel_size=3, stride=1, padding=1,)
        self.conv3_1 = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1,)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,)
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,)
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,)
        self.conv5_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,)
        self.fc = nn.Linear(256, 10)

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

            x = nnf.relu(self.conv1_1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv1_2(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(x, (2, 2))

            x = nnf.relu(self.conv2_1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv2_2(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(x, (2, 2))

            x = nnf.relu(self.conv3_1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv3_2(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv3_3(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(x, (2, 2))

            x = nnf.relu(self.conv4_1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv4_2(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv4_3(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(x, (2, 2))

            x = nnf.relu(self.conv5_1(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv5_2(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.relu(self.conv5_3(x))
            x = nnf.dropout(x, 0.5, training=True)
            x = nnf.max_pool2d(x, (2, 2))

            x = x.view(-1, 256)

            x = self.fc(x)

            out[i] = x

        return out
