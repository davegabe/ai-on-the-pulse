#!/bin/bash

# Change the current directory to the directory of the script
cd "$(dirname "$0")"

# Train the model
python main.py --retrain