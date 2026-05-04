#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""
    data_path: str = 'path/to/your/data.csv'
    heavy_chain_col: str = 'Heavy_Chain_Sequence'
    light_chain_col: str = 'Light_Chain_Sequence'
    payload_col: str = 'payload_smiles'
    linker_col: str = 'linker_smiles'
    scaffold_col: str = 'Linker_scaffold'
    split_col: str = 'split'
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration"""
    molt5_model_path: str = 'path/to/molt5-base'

    # Antibody encoder (AntibodyResidueGAT)
    antibody_hidden_dim: int = 128
    antibody_output_dim: int = 256
    antibody_num_layers: int = 3
    antibody_num_heads: int = 4
    antibody_dropout: float = 0.1
    antibody_max_seq_len: int = 200
    antibody_local_window: int = 2

    # Payload encoder (GATPayloadEncoder)
    gat_input_dim: int = 6
    gat_hidden_dim: int = 128
    gat_output_dim: int = 256
    gat_num_layers: int = 3
    gat_num_heads: int = 4
    gat_dropout: float = 0.1

    # Fusion
    fusion_dim: int = 768
    fusion_dropout: float = 0.1
    use_cross_attention: bool = True

    encoder_noise_std: float = 0.001
    max_length: int = 512


@dataclass
class TrainConfig:
    """Training configuration (template - adjust for your dataset)"""
    epochs: int = 15
    batch_size: int = 12
    learning_rate: float = 5e-5  # Adjust based on your data size
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 3

    eval_interval: int = 5
    eval_num_samples: int = 200
    eval_num_sequences: int = 20

    use_label_smoothing: bool = True
    label_smoothing: float = 0.1

    output_dir: str = './models/molt5_v9'
    resume_from_checkpoint: str = None


@dataclass
class GenerationConfig:
    """Generation configuration"""
    num_return_sequences: int = 30
    temperature: float = 0.85
    top_p: float = 0.90
    top_k: int = 40
    repetition_penalty: float = 1.05
    min_length: int = 5
    max_length: int = 128
    num_beams: int = 1


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()
