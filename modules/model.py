import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Respeller(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder

        self.decoder

        self.quantiser

    def forward(self, inputs):
        encodings = self.encoder(inputs)
        logits = self.decoder(encodings)
        respelling = self.quantiser(logits)
        return respelling
