from argparse import ArgumentParser
from pathlib import Path
from model.ptmodel import PTModel
from model.utils import load_checkpoint


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--infos", type=Path, required=True)
    return parser


def display_info(model: PTModel):
    print(
        f"Model Name: '{model.model_configuration.name}'"
        f"\n - weights: {model.parameters_count:,}"
        f"\n - branch: '{model.branch_infos['name']}' ({model.branch_infos['hash']})"
    )


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.infos)
    model = PTModel.from_checkpoint(checkpoint)
    display_info(model)
