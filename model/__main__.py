from argparse import ArgumentParser
from pathlib import Path
from model.ptmodel import PTModel
from model.utils import load_checkpoint
import torch
import torch.nn.functional as F


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
    # compute and display cosine similarity
    button_role, counterpart_role = model.model_state_dict["refpoints_embed.weight"]
    similarity = F.cosine_similarity(button_role, counterpart_role, dim=0)
    angle = torch.rad2deg(torch.acos(similarity.clamp(-1, 1)))
    distance = torch.norm(button_role - counterpart_role, p=2)
    print(f" - role embedding cosine similarity: similarity={similarity:.4f} ; angle={angle:.2f}° ; distance={distance:.2f}")


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.infos, device=torch.device("cpu"))
    model = PTModel.from_checkpoint(checkpoint)
    display_info(model)
