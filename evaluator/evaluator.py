from .metric import Metric


class Evaluator:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

    def __call__(self, pred_b_boxes, pred_c_boxes, pred_scores, gt_b_boxes, gt_c_boxes):
        """
        
        """
        results = Metric(self.metric_name)
