from train.train import train
from model.config import ModelConfig
from pathlib import Path

if __name__ == "__main__":
    import argparse
    # init the arguments parser and get the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, default="model.json")
    parser.add_argument("--save-weights", type=str, default="checkpoints")
    args = parser.parse_args()
    # get the model configuration
    config_path = Path(args.model)
    model_config = ModelConfig.open(config_path)

    train(
        model_config,
        resume_path=args.resume,
        save_weights_folder=args.save_weights,
    )
