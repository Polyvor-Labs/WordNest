# config.py

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

# Path to save models
MODEL_SAVE_PATH = "./dist/wordnest_model.pth"