"""PMF: Pooling-based Multimodal Fusion with polarity/magnitude heads."""
import torch
from torch import nn

from ..subNets.PolMagHead import PolMagHead, pack_prediction


class PMF(nn.Module):
    def __init__(self, args):
        super().__init__()
        if not args.get("polmag"):
            raise ValueError("PMF requires polmag=True (polarity and magnitude heads).")
        self.feature_dims = tuple(args.feature_dims)
        dim = args.get("projection_dim", 128)
        fusion_dim = args.get("fusion_dim", 128)
        self.standardization_eps = args.get("standardization_eps", 1e-6)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(size, dim), nn.ReLU()) for size in self.feature_dims
        ])
        self.fusion = nn.Sequential(
            nn.Linear(3 * dim, fusion_dim), nn.ReLU(), nn.Dropout(args.get("dropout", 0.2))
        )
        self.polmag_head = PolMagHead(fusion_dim)
        self.register_buffer("feature_mean", torch.zeros(sum(self.feature_dims)))
        self.register_buffer("feature_std", torch.ones(sum(self.feature_dims)))
        self.register_buffer("normalization_fitted", torch.tensor(False))

    @staticmethod
    def pooled_features(text, audio, vision, padding_mask):
        """Pool valid sequence positions, retaining actual missing positions.

        padding_mask is the supplied attention mask (1=valid, 0=padding),
        NOT a mask inferred from zero-valued features or UNK tokens.
        """
        if padding_mask is None or padding_mask.ndim != 2:
            raise ValueError("PMF requires a [batch, time] padding mask.")
        mask = padding_mask.bool()
        count = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = []
        for value in (text, audio, vision):
            if value.ndim != 3 or value.shape[:2] != mask.shape:
                raise ValueError("PMF needs aligned [batch, time, features] tensors matching the mask.")
            value = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))
            pooled.append(value.sum(dim=1) / count)
        return torch.cat(pooled, dim=-1)

    @torch.no_grad()
    def fit_normalization(self, train_pooled):
        """Fit per-feature population statistics on pooled TRAIN examples only."""
        if train_pooled.ndim != 2 or train_pooled.shape[1] != self.feature_mean.numel() or len(train_pooled) == 0:
            raise ValueError("Invalid training features for PMF standardization.")
        if not torch.isfinite(train_pooled).all():
            raise ValueError("Non-finite training features for PMF standardization.")
        values = train_pooled.double()
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False)
        # Constant/near-constant dimensions are centered without amplifying noise.
        std = torch.where(std >= self.standardization_eps, std, torch.ones_like(std))
        self.feature_mean.copy_(mean)
        self.feature_std.copy_(std)
        self.normalization_fitted.fill_(True)

    def forward(self, text, audio, vision, padding_mask=None):
        if not self.normalization_fitted.item():
            raise RuntimeError("Fit PMF normalization on training data or load a fitted checkpoint first.")
        pooled = self.pooled_features(text, audio, vision, padding_mask)
        standardized = (pooled - self.feature_mean) / self.feature_std
        vectors = standardized.split(self.feature_dims, dim=-1)
        features = [projection(value) for projection, value in zip(self.projections, vectors)]
        hidden = self.fusion(torch.cat(features, dim=-1))
        result = pack_prediction(hidden, True, self.polmag_head)
        result.update(Feature_t=features[0], Feature_a=features[1], Feature_v=features[2], Feature_f=hidden)
        return result
