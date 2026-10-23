from evaluate import Evaluator, PairDistanceAllEvaluation, PairIoUAllEvaluation, PairDistanceEvaluation, PairIoUEvaluation


def get_evaluator(threshold: float = 0.9) -> Evaluator:
    """
    Returns an evaluator with the given threshold.
    """
    evaluator = Evaluator()
    evaluator.add_evaluation("distance", PairDistanceAllEvaluation())
    evaluator.add_evaluation("iou", PairIoUAllEvaluation())
    return evaluator