from enum import Enum


class Mode(Enum):
    """
    Describe the inference mode of the model.
    This will impact the output of the model.
    """
    DEBUG = 1 # return all tensors
    TRAINING = 2 # return only predictions
    EVALUATION = 3 # return only predictions