import torch
from collections import Counter

class TextTokenizer:
    """
    TextTokenizer is a simple tokenizer that builds a vocabulary from a given text file.
    It provides methods to encode text into token indices and decode token indices back to text.
    
    Attributes:
        vocab_size (int): The maximum number of tokens in the vocabulary.
        word2idx (dict): A mapping from words to their corresponding token indices.
        idx2word (dict): A mapping from token indices to their corresponding words.
    """
    def __init__(self, file_path=None, vocab_size=10000):
        """
        Initializes the TextTokenizer by building a vocabulary from a text file if provided.
        The vocabulary includes the most common words up to vocab_size (including special tokens).

        Args:
            file_path (str, optional): Path to the text file to build the vocabulary. 
                                       If not provided, an empty vocabulary is initialized.
            vocab_size (int): Maximum vocabulary size including special tokens (default: 10000).
                              Special tokens include '<unk>' for unknown words and '<pad>' for padding.
        """
        self.vocab_size = vocab_size
        if file_path:
            # Read the text file and convert all text to lowercase.
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().lower()

            # Split the text into words.
            words = text.split()
            # Count the frequency of each word.
            word_counts = Counter(words)
            # Select the most common words, reserving one slot for the '<unk>' token.
            common_words = word_counts.most_common(vocab_size - 1)

            # Initialize the vocabulary with special tokens.
            self.word2idx = {'<unk>': 0, '<pad>': 1}
            self.idx2word = {0: '<unk>', 1: '<pad>'}

            # Add the common words to the vocabulary.
            for idx, (word, _) in enumerate(common_words, 2):
                self.word2idx[word] = idx
                self.idx2word[idx] = word

            # Ensure that the vocabulary size does not exceed the specified maximum.
            self.vocab_size = min(len(self.word2idx), vocab_size)
        else:
            # If no file is provided, initialize empty mappings.
            self.word2idx = {}
            self.idx2word = {}

    def encode(self, text):
        """
        Encodes a text string into a list of token indices using the built vocabulary.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: A list of token indices corresponding to the words in the text.
                  Words not found in the vocabulary are mapped to the index for '<unk>'.
        """
        # Convert text to lowercase, split into words, and map each word to its index.
        return [self.word2idx.get(word, 0) for word in text.lower().split()]
    
    def decode(self, tokens):
        """
        Decodes a list of token indices back into a text string.

        Args:
            tokens (list): A list of token indices.

        Returns:
            str: The decoded text string, with '<pad>' tokens removed.
        """
        # Map each token index to its corresponding word, ignoring padding tokens.
        return ' '.join([self.idx2word.get(token, '<unk>') for token in tokens if token != 1])

    def load_vocab(self, vocab_path):
        """
        Loads a pre-saved vocabulary from a file.

        Args:
            vocab_path (str): The path to the file containing the saved vocabulary.
                              The file should be a torch file with a dictionary containing 
                              'word2idx' and 'idx2word' keys.
        """
        vocab = torch.load(vocab_path)
        self.word2idx = vocab['word2idx']
        self.idx2word = vocab['idx2word']
        self.vocab_size = len(self.word2idx)