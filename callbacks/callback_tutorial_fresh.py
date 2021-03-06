"""
Callback Extension Example
================================

This example should illustrate how to extend the training
using simple callbacks. In particular we will modulate the learning rate
with a sawtooth function and clip the gradients by value
"""

##################################################
# change directories to your needs
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.utils.python_utils import ensure_dir
LOG_DIRECTORY = ensure_dir('log/sawtooth')
SAVE_DIRECTORY = ensure_dir('save')
DATASET_DIRECTORY = ensure_dir('dataset')
print("\n \n LOGDIR", LOG_DIRECTORY)

##################################################
# shall models be downloaded
DOWNLOAD_CIFAR = True
USE_CUDA = True

##################################################
# Build torch model
import torch.nn as nn
from inferno.extensions.layers import ConvELU2D
from inferno.extensions.layers import Flatten
model = nn.Sequential(
    ConvELU2D(in_channels=3, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(in_features=(256 * 4 * 4), out_features=10),
    nn.Softmax()
)


##################################################
# data loaders
from inferno.io.box.cifar import get_cifar10_loaders
train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
                                                    download=DOWNLOAD_CIFAR)

logger = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                           log_images_every='never')


##################################################
# Build trainer
from inferno.trainers.basic import Trainer

trainer = Trainer(model)
trainer.build_criterion('CrossEntropyLoss')
trainer.build_metric('CategoricalError')
trainer.build_optimizer('Adam')
trainer.validate_every((2, 'epochs'))
trainer.save_every((5, 'epochs'))
trainer.save_to_directory(SAVE_DIRECTORY)
trainer.set_max_num_epochs(10)



trainer.build_logger(logger,
                     log_directory=LOG_DIRECTORY)
trainer.set_log_directory(LOG_DIRECTORY)


##################################################
# Bind loaders
trainer.bind_loader('train', train_loader)
trainer.bind_loader('validate', validate_loader)


##################################################

from inferno.trainers.callbacks.base import Callback

class SawtoothLearningrate(Callback):
    """oscillating learning rate"""
    def __init__(self, min_value, max_value, frequency):
        super(SawtoothLearningrate, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.frequency = frequency
        
    def begin_of_training_iteration(self, **_):

        current_iteration = self.trainer.iteration_count
        sawtooth = (current_iteration % self.frequency) / self.frequency

        new_lr = self.min_value + (self.max_value - self.min_value) * sawtooth

        for paraam_group in self.trainer.optimizer.param_groups:
            paraam_group["lr"] = new_lr

trainer.register_callback(SawtoothLearningrate(0.001, 0.1, 100))

##################################################
# activate cuda
if USE_CUDA:
    trainer.cuda()

##################################################
# fit
trainer.fit()
