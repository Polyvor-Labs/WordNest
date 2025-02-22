import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Wordnest, TextDataset, TextTokenizer

def train_model(model, dataloader, tokenizer, epochs=10, save_path='model.pth', vocab_path='vocab.pth'):
    """
    Trains the Wordnest model using the provided dataloader and tokenizer.
    
    This function performs training for a specified number of epochs using the AdamW optimizer and
    cross-entropy loss. It computes two types of losses: the main loss from the primary output logits
    and the auxiliary MTP loss from additional prediction outputs. The total loss is a combination
    of these two losses, with the MTP loss scaled by a factor of 0.3. The trained model and vocabulary
    are saved at the end of training.
    
    Args:
        model (nn.Module): The Wordnest model to train.
        dataloader (DataLoader): DataLoader providing batches of (inputs, targets).
        tokenizer (TextTokenizer): Tokenizer instance, used for saving the vocabulary.
        epochs (int): Number of training epochs (default: 10).
        save_path (str): File path to save the trained model parameters.
        vocab_path (str): File path to save the tokenizer vocabulary.
    """
    # Determine the device to run on (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up the optimizer and loss criterion
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Use CrossEntropyLoss; ignore padding token with index 1
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    
    # Loop over epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # Accumulate loss over batches
        
        # Iterate over batches from the dataloader
        for inputs, targets in dataloader:
            # Move inputs and targets to the selected device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients before the backward pass
            optimizer.zero_grad()
            # Forward pass: obtain main logits and MTP logits from the model
            main_logits, mtp_logits = model(inputs, targets)
            
            # Compute main loss by flattening the logits and targets
            loss_main = criterion(
                main_logits.view(-1, main_logits.size(-1)),
                targets.view(-1)
            )
            
            # Initialize MTP loss and accumulate loss over MTP outputs
            loss_mtp = 0.0
            for i, logits in enumerate(mtp_logits):
                # Shift targets accordingly for the MTP module
                mtp_targets = targets[:, i+1:i+1+logits.size(1)]
                loss_mtp += criterion(
                    logits.view(-1, logits.size(-1)),
                    mtp_targets.contiguous().view(-1)
                )
            
            # Combine main loss and MTP loss (scale MTP loss by 0.3); if no MTP outputs, use main loss
            batch_loss = loss_main + (0.3 * (loss_mtp / len(mtp_logits)) if mtp_logits else 0)
            batch_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Accumulate the batch loss (using .item() to get the scalar value)
            running_loss += batch_loss.item()
        
        # Compute average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    # Save the trained model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')
    
    # Save the tokenizer vocabulary (word2idx and idx2word)
    torch.save({
        'word2idx': tokenizer.word2idx,
        'idx2word': tokenizer.idx2word
    }, vocab_path)
    print(f'Vocab saved to {vocab_path}')

if __name__ == "__main__":
    # Configuration parameters
    txt_file = "datasets/corpus_data.txt"   # Path to the training text corpus
    seq_length = 256                          # Length of each training sequence
    batch_size = 2                            # Batch size for training
    epochs = 20                               # Number of training epochs
    vocab_size = 10000                        # Maximum vocabulary size
    save_path = 'dist/wordnest_model.pth'           # File path to save the trained model
    vocab_path = 'dist/vocab.pth'                  # File path to save the vocabulary
    
    # Initialize the tokenizer with the text file and vocabulary size
    tokenizer = TextTokenizer(txt_file, vocab_size)
    # Create the dataset using the text file, tokenizer, and sequence length
    dataset = TextDataset(txt_file, tokenizer, seq_length)
    # Create a DataLoader for batching and shuffling the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the Wordnest model with the specified architecture parameters
    model = Wordnest(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        n_layers=6,
        n_experts=8,
        mtp_depth=1
    )
    
    # Start the training process
    train_model(model, dataloader, tokenizer, epochs, save_path, vocab_path)