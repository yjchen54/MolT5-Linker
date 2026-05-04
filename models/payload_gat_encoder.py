#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""GAT Payload Encoder for molecular structure encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

try:
    from torch_geometric.nn import GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    Data = Batch = None


def smiles_to_graph(smiles: str):
    """Convert SMILES to graph with 6D node features."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 100.0,
                atom.GetDegree() / 4.0,
                atom.GetFormalCharge(),
                atom.GetTotalNumHs() / 4.0,
                float(atom.GetIsAromatic()),
                float(atom.IsInRing())
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)
    except:
        return None


class GATPayloadEncoder(nn.Module):
    """Graph attention network for payload encoding."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4
    ):
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=num_heads, concat=True))

        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, concat=True))

        self.gat_layers.append(GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False))

    def forward(self, smiles_list):
        """Encode payload SMILES."""
        graphs = [smiles_to_graph(s) for s in smiles_list]

        valid_graphs = [g for g in graphs if g is not None]
        if not valid_graphs:
            device = next(self.parameters()).device
            return torch.zeros(len(smiles_list), self.output_dim, device=device)

        device = next(self.parameters()).device
        batch = Batch.from_data_list(valid_graphs).to(device)

        x = batch.x
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, batch.edge_index)
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)

        out = global_mean_pool(x, batch.batch)

        result = torch.zeros(len(smiles_list), self.output_dim, device=device)
        valid_idx = 0
        for i, g in enumerate(graphs):
            if g is not None:
                result[i] = out[valid_idx]
                valid_idx += 1

        return result
