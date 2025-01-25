import torch
from models import WordNestLanguageModel, AutoregressiveWrapper, Tokenizer, get_device
from config import DIM, HEADS, HEAD_DIM, FF_DIM, NUM_LAYERS, NUM_EXPERTS, MAX_SEQ_LEN, MAX_LEN, DROPOUT, TEMPERATURE, LEARNING_RATE, MODEL_SAVE_PATH
from torchinfo import summary

# Define model parameters (aligned with the main model code)
params = {
    'num_tokens': 65,  # Default tokenizer size (65 characters)
    'dim': DIM,
    'heads': HEADS,
    'head_dim': HEAD_DIM,
    'ff_dim': FF_DIM,
    'num_layers': NUM_LAYERS,
    'max_seq_len': MAX_SEQ_LEN,
    'dropout': DROPOUT,
    'num_experts': NUM_EXPERTS,  # Number of experts in MoE
    'dict_path': 'tokenizer_dict.json'
}

# Initialize Tokenizer
tokenizer = Tokenizer(dict_path=params['dict_path'])

# Verify tokenizer size matches model's num_tokens
assert tokenizer.size() == params['num_tokens'], \
    f"Tokenizer size ({tokenizer.size()}) does not match model's num_tokens ({params['num_tokens']})"

# Initialize Model with MoE and AutoregressiveWrapper
model = WordNestLanguageModel(
    num_tokens=params['num_tokens'],
    dim=params['dim'],
    heads=params['heads'],
    head_dim=params['head_dim'],
    ff_dim=params['ff_dim'],
    num_layers=params['num_layers'],
    max_seq_len=params['max_seq_len'],
    dropout=params['dropout'],
    num_experts=params['num_experts']
)
model = AutoregressiveWrapper(model)  # Wrap model before moving to device
model.to(get_device())  # Move model to the appropriate device

# Generate random input tensor on the correct device
input_tensor = torch.randint(0, params['num_tokens'], (1, params['max_seq_len']), device=get_device())

# Model summary using torchinfo
summary(
    model.model,  # Access the main model inside AutoregressiveWrapper
    input_data=input_tensor,
    depth=5,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    row_settings=["var_names"]
)

# Print model parameters
print("\nModel Parameters:")
print(f"Number of Tokens: {params['num_tokens']}")
print(f"Embedding Dimension: {params['dim']}")
print(f"Number of Heads: {params['heads']}")
print(f"Head Dimension: {params['head_dim']}")
print(f"Feedforward Dimension: {params['ff_dim']}")
print(f"Number of Layers: {params['num_layers']}")
print(f"Maximum Sequence Length: {params['max_seq_len']}")
print(f"Dropout Rate: {params['dropout']}")
print(f"Number of Experts (MoE): {params['num_experts']}")

# Print tokenizer parameters
print("\nTokenizer Parameters:")
print(f"Vocabulary Size: {tokenizer.size()}")

print("\nParameter check completed.")