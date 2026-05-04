#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MolT5 with Dual-Branch GAT Fusion
Combines antibody and payload encoders for ADC linker generation.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from typing import Optional, List, Tuple

from .antibody_gat_encoder import AntibodyResidueGAT
from .gat_encoder import GATPayloadEncoder


class MolT5WithGATFusion(nn.Module):
    """ADC linker generator with dual-branch GAT fusion."""

    def __init__(
        self,
        molt5_model_path: str,
        antibody_config: dict,
        gat_config: dict,
        fusion_dim: int = 768,
        noise_std: float = 0.01
    ):
        super().__init__()
        self.noise_std = noise_std

        ab_out_dim = antibody_config.get('output_dim', 256)
        self.antibody_encoder = AntibodyResidueGAT(
            hidden_dim=antibody_config.get('hidden_dim', 128),
            output_dim=ab_out_dim,
            num_layers=antibody_config.get('num_layers', 3),
            num_heads=antibody_config.get('num_heads', 4),
            dropout=antibody_config.get('dropout', 0.1),
            max_seq_len=antibody_config.get('max_seq_len', 200),
            local_window=antibody_config.get('local_window', 2)
        )

        pay_out_dim = gat_config['output_dim']
        self.payload_encoder = GATPayloadEncoder(
            input_dim=gat_config['input_dim'],
            hidden_dim=gat_config['hidden_dim'],
            output_dim=pay_out_dim,
            num_layers=gat_config['num_layers'],
            num_heads=gat_config['num_heads']
        )

        fuse_in = ab_out_dim + pay_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        self.molt5 = T5ForConditionalGeneration.from_pretrained(molt5_model_path)
        self.num_prefix_tokens = 4
        self.prefix_projection = nn.Linear(fusion_dim, self.molt5.config.d_model * self.num_prefix_tokens)

    def encode_context(
        self,
        heavy_chains: List[str],
        light_chains: List[str],
        payload_smiles_list: List[str]
    ) -> torch.Tensor:
        """Encode antibody and payload into prefix tokens."""
        h_ab = self.antibody_encoder(heavy_chains, light_chains)
        h_pay = self.payload_encoder(payload_smiles_list)

        fused = self.fusion(torch.cat([h_ab, h_pay], dim=1))

        if self.training and self.noise_std > 0:
            fused = fused + torch.randn_like(fused) * self.noise_std

        prefix_flat = self.prefix_projection(fused)
        batch_size = fused.size(0)
        prefix_tokens = prefix_flat.view(batch_size, self.num_prefix_tokens, self.molt5.config.d_model)

        return prefix_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        heavy_chains: List[str],
        light_chains: List[str],
        payload_smiles_list: List[str],
        labels: Optional[torch.Tensor] = None
    ):
        """Forward pass with context encoding."""
        prefix_tokens = self.encode_context(heavy_chains, light_chains, payload_smiles_list)

        encoder_outputs = self.molt5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        fused_hidden_states = torch.cat([prefix_tokens, encoder_outputs.last_hidden_state], dim=1)

        batch_size = input_ids.size(0)
        device = input_ids.device
        prefix_attention_mask = torch.ones(
            batch_size, self.num_prefix_tokens,
            dtype=attention_mask.dtype,
            device=device
        )
        extended_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        from transformers.modeling_outputs import BaseModelOutput
        expanded_encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden_states)

        outputs = self.molt5(
            encoder_outputs=expanded_encoder_outputs,
            attention_mask=extended_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs
