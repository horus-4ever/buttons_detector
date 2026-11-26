from argparse import ArgumentParser
from pathlib import Path
from model.prtr import PRTR, load_model_from_weights
from model.ptmodel import PTModel
from model.utils import load_checkpoint
import torch
import torch.nn.functional as F


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--infos", type=Path, required=True)
    return parser


def get_general_informations(model: PTModel):
    return {
        "weights": f"{model.parameters_count:,}",
        "branch name": model.branch_infos["name"],
        "branch hash": model.branch_infos["hash"]
    }

def get_roleembed_informations(model: PTModel):
    # compute and display cosine similarity
    button_role, counterpart_role = model.model_state_dict["refpoints_embed.weight"]
    similarity = F.cosine_similarity(button_role, counterpart_role, dim=0)
    angle = torch.rad2deg(torch.acos(similarity.clamp(-1, 1)))
    distance = torch.norm(button_role - counterpart_role, p=2)
    return {
        "cosine similarity": f"{similarity:.4f}",
        "angle": f"{angle:.2f}°",
        "distance": f"{distance:.2f}"
    }

def get_refpoints_informations(model: PTModel, runner: PRTR):
    # computes the spatial entropy
    object_embed, object_queries = torch.split(runner.query_embed.weight, runner.d_model, 1)
    # [Q, 2]
    reference_points = runner.transformer.proj_reference_points(object_embed).sigmoid()
    # points = []
    # for posx, posy in reference_points:
    #     points.append((posx.item(), posy.item()))
    # with open("model/REFPOINTS.py", "w") as file:
    #     print("REFPOINTS =", points, file=file)
    return {}



def get_informations(model: PTModel, runner: PRTR):
    return {
        "General Informations": get_general_informations(model),
        "Role Embedding": get_roleembed_informations(model),
        "Reference Points": get_refpoints_informations(model, runner)
    }


def display_info(model: PTModel, runner: PRTR):
    print(f"Model '{model.model_configuration.name}'")
    informations = get_informations(model, runner)
    for category_name, infos in informations.items():
        print(category_name.capitalize())
        for elem, value in infos.items():
            print(f"  - {elem}: {value}")


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.infos, device=torch.device("cpu"))
    model = PTModel.from_checkpoint(checkpoint)
    runner = load_model_from_weights(args.infos, device=torch.device("cpu"))
    display_info(model, runner)
