import os

def main():
    # Get parent path
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Save as environment variable for next scripts
    os.environ["BASE_PATH"] = script_path
    os.environ["VERSION"] = "v1"
    os.environ["WANDB_USER"] = "your_wandb_username"

    # Run UniTS
    os.system("./src/models/UniTS/run.sh")

    # Run UniTS with subsample
    os.system("./src/models/UniTS/run_subsample.sh")
    
    # Run HypAD
    os.system("./src/models/HypAD/run.sh")

    # Run TranAD
    os.system("./src/models/TranAD/run.sh")
    
if __name__ == "__main__":
    main()