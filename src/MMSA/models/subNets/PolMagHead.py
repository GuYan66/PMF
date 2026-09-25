"""Shared polarity and magnitude heads.

Polarity is a 3-way classifier: negative, neutral, positive.
Magnitude is an absolute intensity in [0, 3]. Signed intensity is 0 when the
predicted class is neutral, and +/- magnitude otherwise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolMagHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.polarity = nn.Linear(in_dim, 3)
        self.magnitude = nn.Linear(in_dim, 1)

    def forward(self, hidden):
        logits = self.polarity(hidden)
        magnitude = 3.0 * torch.sigmoid(self.magnitude(hidden)).view(-1)
        return logits, magnitude


def polarity_from_intensity(y):
    """Ground-truth intensity to negative=0, neutral=1, positive=2."""
    labels = torch.ones(y.shape, dtype=torch.long, device=y.device)
    labels = torch.where(y < 0, torch.zeros_like(labels), labels)
    labels = torch.where(y > 0, torch.full_like(labels, 2), labels)
    return labels


def compose_intensity(polarity, magnitude):
    signed = torch.where(polarity == 0, -magnitude, magnitude)
    signed = torch.where(polarity == 1, torch.zeros_like(signed), signed)
    return signed


def pack_prediction(hidden, polmag, head, linear=None, scale_regression=False):
    """Map a fused vector to the dict trainers expect.

    Regression keeps the original linear head. polmag replaces it.
    """
    if polmag:
        logits, magnitude = head(hidden)
        intensity = compose_intensity(logits.argmax(dim=-1), magnitude)
        return {
            "polarity": logits,
            "magnitude": magnitude,
            "M": intensity.unsqueeze(-1),
        }
    output = linear(hidden)
    if scale_regression:
        output = torch.sigmoid(output) * 6.0 - 3.0
    return {"M": output}
