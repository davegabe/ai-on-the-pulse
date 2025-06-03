#!/bin/bash

# Change the current directory to the directory of the script
cd "$(dirname "$0")"

# Train the model
python main.py -c wesad.yaml

python main.py -c dreamer.yaml

python main.py -c hci.yaml