from evaluate import Evaluator, PairDistanceAllEvaluation, PairIoUAllEvaluation, EvaluationGroup


def get_groups() -> list[EvaluationGroup]:
    return [
        EvaluationGroup.INVISIBLE_GROUP(),
        EvaluationGroup.VISIBLE_GROUP()
    ]



def get_evaluator(threshold: float = 0.9) -> Evaluator:
    """
    Returns an evaluator with the given threshold.
    """
    groups = get_groups()
    evaluator = Evaluator()
    evaluator.add_evaluation("distance", PairDistanceAllEvaluation(groups=groups))
    evaluator.add_evaluation("iou", PairIoUAllEvaluation(groups=groups))
    return evaluator
