# System
import os
import sys
# Train model
from train.train_model import run_train_model

# Seed
seed = 42

# Get the arguments from the command-line
argv = sys.argv[1:]
abs_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(abs_path, 'data', argv[0]) # blogtext.csv.zip

# Run training & save best model to the blobs folder
run_train_model(abs_path, path, seed)