import torch
import torch.nn.functional as F
import argparse
from models import Wordnest, TextTokenizer

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """
    Generates text using the provided language model and tokenizer.

    This function takes an input prompt, tokenizes it, and iteratively generates new tokens
    by sampling from the model's output distribution. The generation continues for a maximum
    of `max_length` tokens or until some stopping criteria are met.

    Args:
        model (nn.Module): The trained language model (Wordnest) used for text generation.
        tokenizer (TextTokenizer): Tokenizer used for encoding input text and decoding token sequences.
        prompt (str): The initial text prompt to start generation.
        max_length (int): Maximum number of tokens to generate (default: 50).
        temperature (float): Temperature parameter for controlling randomness in sampling (default: 0.7).

    Returns:
        str: The generated text as a decoded string.
    """
    # Set the model to evaluation mode
    model.eval()
    # Retrieve the device from the model parameters (CPU or GPU)
    device = next(model.parameters()).device
    
    # Encode the prompt into token indices using the tokenizer
    input_tokens = tokenizer.encode(prompt)
    # Convert tokens to a tensor and add a batch dimension, then move to the appropriate device
    input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)
    
    # Copy the input tokens to initialize the generated token list
    generated = input_tokens.copy()
    
    # Loop to generate each token up to max_length
    for _ in range(max_length):
        with torch.no_grad():
            # Forward pass through the model to obtain logits (predictions) for the next token
            logits, _ = model(input_tensor, None)
        
        # Retrieve logits for the last token in the sequence, then adjust with temperature
        logits = logits[0, -1, :] / temperature
        # Compute probabilities using softmax over the logits
        probs = torch.softmax(logits, dim=-1)
        # Sample the next token from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the generated token to the list
        generated.append(next_token.item())
        
        # Prepare the next token for concatenation by adding the batch dimension
        next_token = next_token.unsqueeze(0)  # Shape: [1, 1]
        # Append the new token to the current input tensor for the next iteration
        input_tensor = torch.cat([input_tensor, next_token], dim=1)
        
        # If the input tensor becomes too long, truncate it to the last 512 tokens
        if input_tensor.size(1) > 512:
            input_tensor = input_tensor[:, -512:]
    
    # Decode the generated token sequence back into text and return it
    return tokenizer.decode(generated)

if __name__ == "__main__":
    # Set up the argument parser for command line options
    parser = argparse.ArgumentParser(description="Generate text using Wordnest model")
    parser.add_argument("--prompt", type=str, default="Hello", help="Text prompt for generation")
    parser.add_argument("--model_path", type=str, default="dist/wordnest_model.pth", help="Path to the trained model file")
    parser.add_argument("--vocab_path", type=str, default="dist/vocab.pth", help="Path to the vocabulary file")
    args = parser.parse_args()
    
    # Load the tokenizer and the vocabulary from the specified file
    tokenizer = TextTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    
    # Initialize the Wordnest model with the correct vocabulary size and architecture parameters
    model = Wordnest(vocab_size=tokenizer.vocab_size, d_model=512, n_layers=6, n_experts=8, mtp_depth=1)
    # Load the trained model parameters from file
    model.load_state_dict(torch.load(args.model_path))
    print(f'Model loaded from {args.model_path}')
    
    # Generate text based on the provided prompt
    generated = generate_text(model, tokenizer, args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated}")
