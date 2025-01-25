# WordNest Language Model

WordNest is an advanced Transformer-based language model with **Mixture-of-Experts** (MoE) architecture, designed for high-quality text generation. It combines cutting-edge techniques including **Rotary Positional Embeddings**, **SwiGLU activation**, and **Memory-Augmented Attention** with dynamic expert routing.


![Image](https://i.ibb.co.com/z8G8tqB/logo-wordnest.jpg)

## Features

- **Mixture-of-Experts (MoE) Architecture**: 
  - 4-8 specialized expert networks per layer
  - Dynamic token routing with learned gating
  - Efficient computation through expert parallelism
  
- **Enhanced Components**:
  - Rotary Positional Embeddings (RoPE)
  - Memory-Augmented Attention
  - SwiGLU Activation Function
  - Autoregressive Generation with Temperature Control

## Installation

### Requirements
- Python 3.11+
- PyTorch 2.0+
- Libraries: `numpy`, `tqdm`, `torchinfo`

```bash
git clone https://github.com/Polyvor-Labs/WordNest.git
cd WordNest
pip install -r requirements.txt
```

## Usage
### Training the Model
Prepare your dataset in .txt format. Each line in the file should contain one sentence or paragraph.

Configure hyperparameters in the config.py file as needed.

Run the training script:

```bash
python train.py \
  --train_file dataset/train.txt \
  --val_file dataset/val.txt \
  --save_path models/wordnest_model.pth \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 0.0003
```

### Generating Text
You can use the trained model to generate text by running the generate.py script:

Indonesia :

```bash
python generate.py --prompt "gol" --model_path dist/id_wordnest_model.pth
```

Example output:

```
gol spektakuler tercipta di menit terakhir pertandingan...
```

English :

```bash
python generate.py --prompt "A spectacular goal" --model_path dist/en_wordnest_model.pth
```

Example output:

```
A spectacular goal was scored in the last minute of the match after...
```

### Hyperparameters
Hyperparameters can be configured in the config.py file. Some important parameters include:

```py
# Hyperparameter for models
DIM = 512
HEADS = 8
HEAD_DIM = 64
FF_DIM = 1024
NUM_LAYERS = 6
MAX_SEQ_LEN = 1024
DROPOUT = 0.1
NUM_EXPERTS = 2

# Hyperparameter for training
LEARNING_RATE = 0.0003

# Hyperparameter for generate
MAX_LEN = 50
TEMPERATURE = 0.7

```

# Model Parameter Inspection
To inspect the model's parameters, and architecture, use the param.py script. This script provides a detailed breakdown of the model's components, including the Mixture-of-Experts (MoE) layers.

Usage
Run the following command to inspect the model:

```bash
python param.py
```

Output Example :
The script will display the following information:

```
====================================================================================================================================================================================
Layer (type (var_name))                                 Input Shape               Output Shape              Param #                   Kernel Shape              Mult-Adds
====================================================================================================================================================================================
WordNestLanguageModel (WordNestLanguageModel)           [1, 1024]                 [1, 1024, 65]             --                        --                        --
├─Embedding (token_embedding)                           [1, 1024]                 [1, 1024, 512]            33,280                    --                        33,280
├─RotaryPositionalEmbedding (rope)                      [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
├─ModuleList (layers)                                   --                        --                        --                        --                        --
│    └─DecoderLayerWithMoE (0)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    └─DecoderLayerWithMoE (1)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    └─DecoderLayerWithMoE (2)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    └─DecoderLayerWithMoE (3)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    └─DecoderLayerWithMoE (4)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    └─DecoderLayerWithMoE (5)                          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (attention)                         [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MemoryAugmentedAttention (fn)          [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (to_qkv)                   [1, 1024, 512]            [1, 1024, 1536]           786,432                   --                        786,432
│    │    │    │    └─Linear (to_out)                   [1, 1024, 512]            [1, 1024, 512]            262,656                   --                        262,656
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    └─PreNorm (moe)                               [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    └─LayerNorm (norm)                       [1, 1024, 512]            [1, 1024, 512]            1,024                     --                        1,024
│    │    │    └─MixtureOfExperts (fn)                  [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
│    │    │    │    └─Linear (router)                   [1, 1024, 512]            [1, 1024, 2]              1,026                     --                        1,026
│    │    │    │    └─Softmax (softmax)                 [1, 1024, 2]              [1, 1024, 2]              --                        --                        --
│    │    │    │    └─ModuleList (experts)              --                        --                        6,298,624                 --                        --
│    │    └─Dropout (dropout)                           [1, 1024, 512]            [1, 1024, 512]            --                        --                        --
├─Linear (to_logits)                                    [1, 1024, 512]            [1, 1024, 65]             33,345                    --                        33,345
====================================================================================================================================================================================
Total params: 44,171,341
Trainable params: 44,171,341
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 44.17
====================================================================================================================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 508.14
Params size (MB): 176.69
Estimated Total Size (MB): 684.84
====================================================================================================================================================================================

Model Parameters:
Number of Tokens: 65
Embedding Dimension: 512
Number of Heads: 8
Head Dimension: 64
Feedforward Dimension: 1024
Number of Layers: 6
Maximum Sequence Length: 1024
Dropout Rate: 0.1
Number of Experts (MoE): 2

Tokenizer Parameters:
Vocabulary Size: 65
```


# MoE Architecture Overview

```
graph TD
    Input --> TokenEmbedding
    TokenEmbedding --> RoPE
    RoPE --> MoEBlock
    MoEBlock -->|Expert 1| SwiGLU
    MoEBlock -->|Expert 2| SwiGLU
    MoEBlock -->|Expert N| SwiGLU
    SwiGLU --> Router
    Router --> Output
```

# Citation

**WordNest Language Model**
```bibtex
@software{WordNest-Language-Model,
  author = {Zahir Hadi Athallah},
  title = {WordNest: Text Generation Mixture-of-Experts Language Model},
  year = {2025},
  url = {https://github.com/Polyvor-Labs/WordNest},
  note = {A Transformer-based language model with MoE architecture, Rotary Positional Embeddings, and Memory-Augmented Attention.}
}
```

## Contributing
Contributions are welcome! If you'd like to contribute, please open an issue or submit a pull request. Make sure to follow the contribution guidelines.

## License
This project is licensed under the Apache License 2.0.