from models import Tokenizer, WordNestLanguageModel, AutoregressiveWrapper, get_device
from config import DIM, HEADS, HEAD_DIM, FF_DIM, NUM_LAYERS, NUM_EXPERTS, MAX_SEQ_LEN, MAX_LEN, DROPOUT, TEMPERATURE, LEARNING_RATE, MODEL_SAVE_PATH
import torch
import argparse

def load_model(model_path: str):
    """
    Loads the model and tokenizer from the specified path.
    Args:
        model_path (str): Path to the model weights file.
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    # Load tokenizer
    tokenizer = Tokenizer(dict_path='tokenizer_dict.json')
    
    # Initialize the model
    model = WordNestLanguageModel(
        num_tokens=tokenizer.size(),  # Number of tokens in the vocabulary
        dim=DIM,  # Embedding dimension
        heads=HEADS,  # Number of attention heads
        head_dim=HEAD_DIM,  # Dimension of each attention head
        ff_dim=FF_DIM,  # Feedforward dimension
        num_layers=NUM_LAYERS,  # Number of transformer layers
        max_seq_len=MAX_SEQ_LEN,  # Maximum sequence length
        dropout=DROPOUT,  # Dropout rate
        num_experts=NUM_EXPERTS  # Number of experts in the MoE layer
    )
    model = AutoregressiveWrapper(model).to(get_device())  # Wrap the model for autoregressive generation
    
    # Load model weights from the specified path
    model.load_state_dict(torch.load(model_path, map_location=get_device()))
    return model, tokenizer


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Generate text with WordNest model')
    parser.add_argument('--prompt', type=str, default='indonesia', 
                      help='Initial prompt text (default: "indonesia")')
    parser.add_argument('--model_path', type=str, default='dist/en_wordnest_model.pth', 
                      help='Path to the model weights file (default: dist/en_wordnest_model.pth)')
    args = parser.parse_args()

    # Load the model and tokenizer from the specified path
    model, tokenizer = load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")
    
    # Convert the prompt into token IDs
    prompt_tokens = torch.tensor(
        [[tokenizer.character_to_token(c) for c in args.prompt]],  # Tokenize each character in the prompt
        device=get_device()  # Move tokens to the appropriate device (CPU/GPU)
    )
    
    # Generate text using the model
    generated = model.generate(
        prompt=prompt_tokens,  # Input prompt tokens
        max_len=MAX_LEN,  # Maximum length of the generated sequence
        temperature=TEMPERATURE,  # Sampling temperature (controls creativity)
        eos_token=tokenizer.character_to_token('.')  # End-of-sequence token (stops generation when encountered)
    )
    
    # Print the generated text
    print("\nGenerated Text:")
    print(tokenizer.detokenize(generated[0].tolist()))  # Convert token IDs back to text