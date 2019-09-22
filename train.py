import numpy as np
from preprocess import ImagesDS
from utils import accuracy
import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T


class Train:
    num_workers = (8,)
    batch_size = 8
    device = "cuda"

    def __init__(
        self,
        ds: ImagesDS,
        val_ds: ImagesDS,
        model,
        learning_rate=0.0005,
        batch_size=8,
        num_workers=8,
        device="cuda",
    ):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device

        self.loader = D.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        self.val_loader = D.DataLoader(
            ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model

    def fit(self, epochs=10):
        tlen = len(self.loader)
        for epoch in range(epochs):
            tloss = 0
            acc = np.zeros(1)
            for x, y in self.loader:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                target = torch.zeros_like(output, device=self.device)
                target[np.arange(x.size(0)), y] = 1
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                tloss += loss.item()
                acc += accuracy(output.cpu(), y)
                del loss, output, y, x, target
            print(
                "Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%".format(
                    epoch + 1, tloss / tlen, acc[0] / tlen
                )
            )
