#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import torch.nn as nn

from ..layers import BayesianConv2d, BayesianLinear, View


def build_lenet_mnist(in_channel, in_height, in_width, out_labels, **config):
    nmc_train = config['nmc_train']
    nmc_test = config['nmc_test']


    conv1 = BayesianConv2d(in_channels=in_channel,
                       in_height=in_height,
                       in_width=in_width,
                       kernel_size=5,
                       out_channels=6,
                       **config)

    conv2 = BayesianConv2d(in_channels=6,
                       in_height=conv1.out_height//2,
                       in_width=conv1.out_width//2,
                       kernel_size=5,
                       out_channels=16,
                       **config)

    fc1 = BayesianLinear(in_features=(16 * conv2.out_height//2 * conv2.out_width//2),
                       out_features=120,
                       **config)

    fc2 = BayesianLinear(in_features=120,
                       out_features=84,
                       **config)

    fc3 = BayesianLinear(in_features=84,
                       out_features=out_labels,
                       **config)

    lenet = nn.Sequential(
        conv1,
        nn.ReLU(),

        View(-1, -1, 6, conv1.out_height, conv1.out_width),
        nn.MaxPool2d(kernel_size=2),
        View(nmc_train, nmc_test, -1, 6, conv1.out_height//2, conv1.out_width//2),

        conv2,
        nn.ReLU(),

        View(-1, -1, 16, conv2.out_height, conv2.out_width),
        nn.MaxPool2d(kernel_size=2),
        View(nmc_train, nmc_test, -1, 16 * conv2.out_height//2 * conv2.out_width//2),

        fc1,
        nn.ReLU(),

        fc2,
        nn.ReLU(),

        fc3
    )
    return lenet
