#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Antibody GAT Encoder for ADC linker generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = Batch = None

AA_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18,
    'Y': 19, 'U': 20, 'X': 21, '<PAD>': 22, '<START>': 23, '<END>': 24
}

AA_FEATURES = {
    'A': [0.62, 0.0, 0.089, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'C': [0.29, 0.0, 0.121, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'D': [-0.90, -1.0, 0.133, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'E': [-0.74, -1.0, 0.147, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'F': [1.19, 0.0, 0.165, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'G': [0.48, 0.0, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'H': [-0.40, 0.5, 0.155, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'I': [1.38, 0.0, 0.131, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'K': [-1.50, 1.0, 0.146, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    'L': [1.06, 0.0, 0.131, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'M': [0.64, 0.0, 0.149, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    'N': [-0.78, 0.0, 0.132, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'P': [0.12, 0.0, 0.115, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Q': [-0.85, 0.0, 0.146, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'R': [-2.53, 1.0, 0.174, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'S': [-0.18, 0.0, 0.105, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    'T': [-0.05, 0.0, 0.119, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'V': [1.08, 0.0, 0.117, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'W': [0.81, 0.0, 0.204, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'Y': [0.26, 0.0, 0.181, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'U': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

for aa in ['<PAD>', '<START>', '<END>']:
    AA_FEATURES[aa] = [0.0] * 10


class AntibodyResidueGAT(nn.Module):
    """Graph attention network for antibody sequence encoding."""

    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        local_window: int = 2
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.local_window = local_window

        self.aa_embedding = nn.Embedding(len(AA_VOCAB), hidden_dim)

        feat_dim = 10
        self.feat_proj = nn.Linear(feat_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim * 2 if i == 0 else hidden_dim
            self.gat_layers.append(
                GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
            )

        self.site_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def _build_graph(self, heavy: str, light: str):
        """Build residue graph from antibody sequences."""
        seq = heavy + light
        seq = seq[:self.max_seq_len]

        node_features = []
        for aa in seq:
            aa_idx = AA_VOCAB.get(aa, AA_VOCAB['X'])
            emb = self.aa_embedding.weight[aa_idx]
            feat = torch.tensor(AA_FEATURES.get(aa, AA_FEATURES['X']), dtype=torch.float32)
            feat_proj = self.feat_proj(feat)
            node_features.append(torch.cat([emb, feat_proj]))

        x = torch.stack(node_features)

        n = len(seq)
        edges = []
        for i in range(n):
            for j in range(max(0, i - self.local_window), min(n, i + self.local_window + 1)):
                if i != j:
                    edges.append([i, j])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def forward(self, heavy_chains: List[str], light_chains: List[str]) -> torch.Tensor:
        """Encode antibody sequences."""
        device = next(self.parameters()).device
        graphs = [self._build_graph(h, l) for h, l in zip(heavy_chains, light_chains)]
        batch = Batch.from_data_list(graphs).to(device)

        x = batch.x
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, batch.edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        batch_size = len(graphs)
        node_counts = [g.num_nodes for g in graphs]

        outputs = []
        start_idx = 0
        for count in node_counts:
            node_emb = x[start_idx:start_idx + count]

            attn_out, _ = self.site_attention(
                node_emb.unsqueeze(0),
                node_emb.unsqueeze(0),
                node_emb.unsqueeze(0)
            )
            pooled = attn_out.mean(dim=1)
            outputs.append(pooled)
            start_idx += count

        h = torch.cat(outputs, dim=0)
        return self.output_proj(h)

