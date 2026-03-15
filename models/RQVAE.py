from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        codebook = torch.empty(codebook_size, embedding_dim)
        nn.init.uniform_(codebook, -1.0 / codebook_size, 1.0 / codebook_size)

        self.register_buffer("codebook", codebook.clone())
        self.register_buffer("ema_weight", codebook.clone())
        self.register_buffer("ema_count", torch.ones(codebook_size))
        self.last_soft_usage: torch.Tensor | None = None

    def _compute_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        return (
            inputs.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * inputs @ self.codebook.t()
        )

    @torch.inference_mode()
    def _ema_update(self, inputs: torch.Tensor, encodings: torch.Tensor) -> None:
        batch_count = encodings.sum(dim=0)
        batch_weight = encodings.transpose(0, 1) @ inputs

        self.ema_count.mul_(self.decay).add_(batch_count, alpha=1.0 - self.decay)
        self.ema_weight.mul_(self.decay).add_(batch_weight, alpha=1.0 - self.decay)

        total_count = self.ema_count.sum()
        smoothed_count = (
            (self.ema_count + self.eps)
            / (total_count + self.codebook_size * self.eps)
            * total_count
        )
        normalized = smoothed_count.unsqueeze(1).clamp_min(self.eps)
        self.codebook.copy_(self.ema_weight / normalized)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        distances = self._compute_distances(inputs)
        detached_codebook = self.codebook.detach().clone()
        soft_distances = (
            inputs.pow(2).sum(dim=1, keepdim=True)
            + detached_codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * inputs @ detached_codebook.t()
        )
        soft_probs = torch.softmax(
            -soft_distances / max(self.embedding_dim ** 0.5, 1.0), dim=1
        )
        self.last_soft_usage = soft_probs.mean(dim=0)
        codes = torch.argmin(distances, dim=1)
        encodings = F.one_hot(codes, num_classes=self.codebook_size).type_as(inputs)
        quantized = F.embedding(codes, self.codebook)

        if self.training:
            self._ema_update(inputs, encodings)

        return quantized, codes

    @property
    def utilization(self) -> float:
        return float((self.ema_count > 1.0).float().mean().item())


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        codebook_size: int,
        num_layers: int,
        decay: float,
        commitment_cost: float,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.commitment_cost = commitment_cost

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizerEMA(
                    embedding_dim=embedding_dim,
                    codebook_size=codebook_size,
                    decay=decay,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.last_codes: torch.Tensor | None = None
        self.last_avg_residual_norm_per_layer: List[float] = []
        self.last_perplexity_per_layer: List[float] = []
        self.last_soft_usage_per_layer: List[torch.Tensor] = []

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        total_vq_loss = inputs.new_zeros(())

        codes_per_layer = []
        residual_norms = []
        perplexities = []
        soft_usages = []

        for quantizer in self.quantizers:
            quantized, codes = quantizer(residual)
            total_vq_loss = total_vq_loss + self.commitment_cost * F.mse_loss(
                residual, quantized.detach()
            )

            quantized_sum = quantized_sum + quantized
            residual = residual - quantized

            codes_per_layer.append(codes)
            residual_norms.append(float(residual.norm(dim=-1).mean().item()))

            probs = torch.bincount(codes, minlength=quantizer.codebook_size).float()
            probs = probs / probs.sum().clamp_min(1.0)
            perplexity = torch.exp(
                -(probs * (probs + 1e-12).log()).sum()
            )
            perplexities.append(float(perplexity.item()))
            if quantizer.last_soft_usage is not None:
                soft_usages.append(quantizer.last_soft_usage)

        codes_tensor = torch.stack(codes_per_layer, dim=1)
        quantized_st = inputs + (quantized_sum - inputs).detach()

        self.last_codes = codes_tensor.detach()
        self.last_avg_residual_norm_per_layer = residual_norms
        self.last_perplexity_per_layer = perplexities
        self.last_soft_usage_per_layer = soft_usages

        return quantized_st, total_vq_loss, codes_tensor

    @property
    def utilization_per_layer(self) -> List[float]:
        return [quantizer.utilization for quantizer in self.quantizers]


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        codebook_size: int,
        num_layers: int,
        decay: float = 0.99,
        commitment_cost: float = 0.25,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.decay = decay
        self.commitment_cost = commitment_cost

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )
        self.rq = ResidualVectorQuantizer(
            embedding_dim=latent_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            decay=decay,
            commitment_cost=commitment_cost,
        )
        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encoder(inputs)
        quantized, vq_loss, codes = self.rq(latents)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, codes

    @torch.inference_mode()
    def generate_semantic_ids(self, inputs: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        latents = self.encoder(inputs)
        _, _, codes = self.rq(latents)
        if was_training:
            self.train()
        return codes
