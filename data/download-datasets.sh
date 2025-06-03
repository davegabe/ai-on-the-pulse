#!/bin/bash

# Change working directory to the root of the project if i'm in data directory
if [ ! -d "src" ]; then
  cd ..
fi

# Make directory for downloading datasets
mkdir -p data/download
mkdir -p data/raw


#* Download and extract WESAD dataset
if [ ! -f "data/download/WESAD.zip" ]; then
  wget -O data/download/WESAD.zip https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download
  unzip -o data/download/WESAD.zip -d data/raw/
else
  echo "WESAD dataset already downloaded and extracted"
fi

#* Download and extract HCI dataset
if [ ! -f "data/download/HCI.zip" ]; then
  echo "Download the HCI dataset from (https://mahnob-db.eu/hci-tagging/) and place the zip file in the data/download/HCI folder"
else
  # Extract the dataset if it's not already extracted
  if [ ! -d "data/raw/HCI" ]; then
    mkdir -p data/raw/HCI
    unzip -o data/download/HCI.zip -d data/raw/HCI
  else
    echo "HCI dataset already downloaded and extracted"
  fi
fi

#* Download and extract DREAMER dataset
if [ ! -f "data/raw/dreamer/DREAMER.mat" ]; then
  echo "Download the DREAMER dataset from (https://zenodo.org/records/546113) and place the mat file in the data/raw/dreamer folder"
fi