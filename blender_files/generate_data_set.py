import bpy
import os
import sys
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor
import random
import argparse

BLEND_DIR = bpy.path.abspath("//")
BLEND_DIR = os.path.normpath(BLEND_DIR)
if BLEND_DIR not in sys.path:
    sys.path.insert(0, BLEND_DIR)
BLEND_DIR = Path(BLEND_DIR)

from configuration import read_configuration, Parameters, Parameter, SamplingPolicy
import helpers
from scene import Scene, Renderer

CONFIG = read_configuration(BLEND_DIR / "configuration.json")

permutation_parameters, random_parameters = CONFIG.parameters.get_parameters()

def generate_permutations(parameters: Parameters, mode="training"):
    permutation_parameters, random_parameters = parameters.get_parameters()
    permutations = None
    other_parameters = None
    if mode == "training":
        training_permutation_parameters = [param.training_values for param in permutation_parameters]
        permutations = itertools.product(*training_permutation_parameters)
        other_parameters = [param.training_values for param in random_parameters]
    elif mode == "validation":
        validation_permutation_parameters = map(lambda param: param.validation_values, permutation_parameters)
        permutations = itertools.product(*validation_permutation_parameters)
        other_parameters = [param.validation_values for param in random_parameters]
    if permutations is None or other_parameters is None:
        raise ValueError("Permutations are invalid")
    for permutation in permutations:
        random_combination = tuple(random.choice(param) for param in other_parameters)
        permutation = permutation + random_combination
        yield {obj.name: obj for obj in permutation}

def generate_random_all(parameters: Parameters, generate: int, mode="training"):
    params = parameters.parameters
    parameters_values = [param.training_values for param in params]
    generated = list()
    while len(generated) < generate:
        random_combination = tuple(random.choice(param) for param in parameters_values)
        if random_combination in generated:
            continue
        generated.append(random_combination)
        yield {obj.name: obj for obj in random_combination}


def generate_dataset(scene_path: Path, max_generate: int, resume_from: int = 0, mode: str = "training"):
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene_name = scene_path.stem
    scene = Scene.from_current_bpy_context(scene_name)
    renderer = Renderer(scene, BLEND_DIR / "out", count=resume_from)
    count = 0
    for parameters in generate_random_all(CONFIG.parameters, generate=max_generate, mode=mode):
        if count < resume_from:
            count += 1
            continue
        renderer.render_at(parameters, list(range(3, 55)))
        count += 1
    print(count)



if __name__ == "__main__":
    # get command line arguments
    # Blender passes its own args in sys.argv; user args are after "--"!
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Generation of the dataset.")
    parser.add_argument('--mode', dest='generation_mode', default="training", type=str, help='Specify the generation mode ("training" / "validation").')
    parser.add_argument('--resume', dest='resume_from', default=0, type=int, help='Specify where to resume from.')
    args = parser.parse_args(argv)
    # set random seed
    random.seed(CONFIG.seed)
    # then generate the dataset
    for scene_config in CONFIG.scenes:
        scene_path = BLEND_DIR / scene_config.name
        generate_dataset(scene_path, scene_config.max_generate, args.resume_from, args.generation_mode)