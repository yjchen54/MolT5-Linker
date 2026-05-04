# MolT5-Linker

A deep learning model for generating antibody-drug conjugate (ADC) linkers using dual-branch graph attention networks fused with MolT5.

## Overview

This model combines:
- **Antibody Encoder**: Graph attention network on residue-level antibody structure
- **Payload Encoder**: Graph attention network on atom-level payload structure  
- **MolT5 Backbone**: Pre-trained molecular transformer for SMILES generation
- **Cross-Modal Fusion**: Learned fusion of antibody and payload representations

## Architecture

```
Heavy Chain ──┐
              ├──► AntibodyResidueGAT ──┐
Light Chain ──┘                         │
                                        ├──► Fusion ──► MolT5 ──► Linker SMILES
Payload SMILES ──► GATPayloadEncoder ──┘
```

### Key Components

1. **AntibodyResidueGAT**: Encodes antibody sequences as residue graphs with attention to conjugation sites (Lys/Cys)
2. **GATPayloadEncoder**: Encodes payload molecules as atom graphs with functional group awareness
3. **Cross-Modal Fusion**: Projects concatenated embeddings into MolT5 prefix tokens
4. **MolT5 Decoder**: Generates linker SMILES autoregressively

## 🔧 Requirements

```bash
torch>=2.0.0
transformers>=4.30.0
torch-geometric>=2.3.0
rdkit>=2023.3.1
```

## 💻 Usage

### Model Initialization

```python
from models.molt5_fusion_clean import MolT5WithGATFusion
from configs.config_template import get_default_config

config = get_default_config()

# Configure model
antibody_config = {
    'hidden_dim': 128,
    'output_dim': 256,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'max_seq_len': 200,
    'local_window': 2
}

gat_config = {
    'input_dim': 6,
    'hidden_dim': 128,
    'output_dim': 256,
    'num_layers': 3,
    'num_heads': 4
}

model = MolT5WithGATFusion(
    molt5_model_path='path/to/molt5-base',
    antibody_config=antibody_config,
    gat_config=gat_config,
    fusion_dim=768
)
```

### Inference Example

```python
# Prepare inputs
heavy_chain = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS..."
light_chain = "DIQMTQSPSSLSASVGDRVTITCRASQSIS..."
payload_smiles = "O=C1CCC(N1)=O"

# Encode context
prefix_tokens = model.encode_context(
    heavy_chains=[heavy_chain],
    light_chains=[light_chain],
    payload_smiles_list=[payload_smiles]
)

# Generate linker (requires trained weights)
# See paper for generation parameters
```
