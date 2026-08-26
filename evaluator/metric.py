from dataclasses import dataclass


@dataclass
class Metric:
    name: str
    TP: int = 0
    TN: int = 0
    FP: int = 0
    FN: int = 0

    @property
    def precision(self) -> float:
        return float(self.TP) / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        return float(self.TP) / (self.TP + self.FN)

    @property
    def F1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)
