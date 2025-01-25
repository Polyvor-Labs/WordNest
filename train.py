from models import Tokenizer, WordNestLanguageModel, AutoregressiveWrapper, Trainer, get_device
from config import DIM, HEADS, HEAD_DIM, FF_DIM, NUM_LAYERS, NUM_EXPERTS, MAX_SEQ_LEN, DROPOUT, LEARNING_RATE, MODEL_SAVE_PATH
from typing import List
import matplotlib.pyplot as plt
import argparse
import random
import os

def load_data_from_txt(file_path: str) -> List[str]:
    """
    Loads training data from a text file.
    Args:
        file_path (str): Path to the text file containing training data.
    Returns:
        List[str]: A list of strings, where each string is a training sample.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        samples = [line.strip() for line in f if line.strip()]
    return samples


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Train WordNest Language Model')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data .txt file')
    parser.add_argument('--val_file', type=str, help='Path to validation data .txt file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.2, 
                      help='Validation split ratio if no val_file provided (default: 0.2)')
    parser.add_argument('--save_path', type=str, default=MODEL_SAVE_PATH, 
                      help='Path to save the trained model (default: config.MODEL_SAVE_PATH)')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, 
                      help='Learning rate for the optimizer (default: config.LEARNING_RATE)')
    args = parser.parse_args()

    # Initialize tokenizer and model
    tokenizer = Tokenizer()
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
    ).to(get_device())  # Move model to the appropriate device (CPU/GPU)
    model = AutoregressiveWrapper(model)  # Wrap the model for autoregressive training

    # Load training data
    data = load_data_from_txt(args.train_file)
    print(f"Loaded {len(data)} training samples from {args.train_file}")

    # Load validation data or split training data
    if args.val_file:
        val_data = load_data_from_txt(args.val_file)
        print(f"Loaded {len(val_data)} validation samples from {args.val_file}")
    else:
        # Split training data into training and validation sets
        random.shuffle(data)
        split_idx = int(len(data) * (1 - args.val_split))
        val_data = data[split_idx:]
        data = data[:split_idx]
        print(f"Split data into {len(data)} training and {len(val_data)} validation samples")

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        save_path=args.save_path,  # Path to save the trained model
        learning_rate=args.learning_rate  # Learning rate for the optimizer
    )

    # Train the model
    loss, val_loss = trainer.train(
        data=data,  # Training data
        val_data=val_data,  # Validation data
        epochs=args.epochs,  # Number of training epochs
        batch_size=args.batch_size  # Batch size for training
    )

    # Save the trained model and tokenizer dictionary
    trainer.save_model()
    tokenizer.save_dictionary('tokenizer_dict.json')
    print(f"Model saved to {args.save_path}")
    print(f"Training completed with learning rate: {args.learning_rate}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Progress")
    plt.savefig('training_loss.png')  # Save the plot as an image
    plt.show()  # Display the plot