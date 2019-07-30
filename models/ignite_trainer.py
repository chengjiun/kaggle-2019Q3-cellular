import torch
import torch.nn as nn
from torchvision import models as models
from torch.optim.lr_scheduler import ExponentialLR

from ignite.metrics import Loss, Accuracy


class DensenetTrainer:
    def __init__(self, model=models.densenet121, lr=0.001, gamma=0.99, n_classes=1108):
        self.model = model(pretrained=True)
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.metrics = {"loss": Loss(self.criterion), "accuracy": Accuracy()}
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
        self.new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self._replace_classifiers()
        self._use_6_channels()
        self.epoch = 0
        self.loss = 999.0

    def _replace_classifiers(self):
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(num_ftrs, self.n_classes)

    def _use_6_channels(self):
        # let's make our model work with 6 channels
        trained_kernel = self.model.features.conv0.weight
        with torch.no_grad():
            self.new_conv.weight[:, :] = torch.stack(
                [torch.mean(trained_kernel, 1)] * 6, dim=1
            )

        self.model.features.conv0 = self.new_conv

    def save_model(self, checkpoint_file):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            checkpoint_file,
        )
        self.checkpoint_file = checkpoint_file

    def load_model(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.loss = checkpoint["loss"]
        self.checkpoint_file = checkpoint_file


class ResnetTrainer(DensenetTrainer):
    def _replace_classifiers(self):
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, self.n_classes)

    def _use_6_channels(self):
        # let's make our model work with 6 channels
        trained_kernel = self.model.conv1.weight
        with torch.no_grad():
            self.new_conv.weight[:, :] = torch.stack(
                [torch.mean(trained_kernel, 1)] * 6, dim=1
            )
        self.model.conv1 = self.new_conv
