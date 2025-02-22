import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    TextDataset loads text data from a file, tokenizes it, and splits it into sequences 
    for language modeling tasks such as next-token prediction.
    
    Attributes:
        tokenizer: A tokenizer object with an `encode` method that converts text to token IDs.
        seq_length (int): The fixed length of each sequence used for training.
        tokens (list): A list of tokenized integers representing the entire text.
        total_seqs (int): The total number of sequences available in the tokenized text.
    """
    
    def __init__(self, file_path, tokenizer, seq_length=512):
        """
        Initializes the TextDataset by reading the file, converting text to lowercase,
        tokenizing it, and preparing sequences of tokens.
        
        Args:
            file_path (str): Path to the text file.
            tokenizer: Tokenizer with an `encode` method to convert text into tokens.
            seq_length (int): The length of each sequence for training (default: 512).
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Read the file content and convert it to lowercase
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        
        # Tokenize the entire text
        self.tokens = self.tokenizer.encode(text)
        
        # Determine the total number of sequences based on the sequence length
        self.total_seqs = len(self.tokens) // seq_length
    
    def __len__(self):
        """
        Returns:
            int: The total number of sequences available in the dataset.
        """
        return self.total_seqs
    
    def __getitem__(self, idx):
        """
        Retrieves a specific sequence and its corresponding target sequence for training.
        The target sequence is the input sequence shifted by one token.
        
        Args:
            idx (int): Index of the sequence to retrieve.
        
        Returns:
            tuple: A tuple containing:
                - input_tokens (torch.Tensor): Tensor of tokens for the input sequence.
                - target_tokens (torch.Tensor): Tensor of tokens for the target sequence.
        """
        # Calculate the starting and ending indices for the sequence slice.
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        
        # Extract tokens for the sequence and create input and target pairs.
        tokens = self.tokens[start:end]
        # Input: all tokens except the last; Target: all tokens except the first.
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])