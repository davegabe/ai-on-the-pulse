import os
from src.data.WESAD.preprocess import main as preprocess_wesad
from src.data.DREAMER.preprocess import main as preprocess_dreamer
from src.data.HCI.preprocess import main as preprocess_hci


def main():
    # Make preprocessed directory
    os.makedirs("data/preprocessed", exist_ok=True)

    # Preprocess WESAD dataset
    preprocess_wesad()

    # Preprocess DREAMER dataset
    preprocess_dreamer()

    # Preprocess HCI dataset
    preprocess_hci()


if __name__ == "__main__":
    main()
