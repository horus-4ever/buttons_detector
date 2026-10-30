from .reports import EvaluationFigureRequest, LossFigureRequest


def get_validation_reports():
    val_dist_APs = EvaluationFigureRequest(
        [
            "val_evaluation.distance.ALL.AP_1:AP",
            "val_evaluation.distance.ALL.AP_3:AP",
            "val_evaluation.distance.ALL.AP_5:AP",
            "val_evaluation.distance.ALL.AP_10:AP"
        ],
        [
            "AP_1",
            "AP_3",
            "AP_5",
            "AP_10"
        ],
        "Evolution of validation distance APs over the epochs"
    )
    test_dist_APs = EvaluationFigureRequest(
        [
            "test_evaluation.distance.ALL.AP_1:AP",
            "test_evaluation.distance.ALL.AP_3:AP",
            "test_evaluation.distance.ALL.AP_5:AP",
            "test_evaluation.distance.ALL.AP_10:AP"
        ],
        [
            "AP_1",
            "AP_3",
            "AP_5",
            "AP_10"
        ],
        "Evolution of test distance APs over the epochs"
    )
    return {
        "val_dist_APs": val_dist_APs,
        "test_dict_APs": test_dist_APs
    }

def get_loss_reports():
    train_loss_report = LossFigureRequest(
        [
            "train_losses.L1_loss_pair",
            "train_losses.ce_loss",
            "train_losses.GIoU_loss_pair",
            "train_losses.loss"
        ],
        [
            "L1_loss_pair",
            "ce_loss",
            "GIoU_loss_pair",
            "loss"
        ],
        "Evolution of the training losses"
    )
    val_loss_report = LossFigureRequest(
        [
            "val_losses.L1_loss_pair",
            "val_losses.ce_loss",
            "val_losses.GIoU_loss_pair",
            "val_losses.loss"
        ],
        [
            "L1_loss_pair",
            "ce_loss",
            "GIoU_loss_pair",
            "loss"
        ],
        "Evolution of the validation losses"
    )
    val_train_loss_report = LossFigureRequest(
        [
            "train_losses.loss",
            "val_losses.loss"
        ],
        [
            "train loss",
            "validation loss"
        ],
        "Evolution of training and validation losses"
    )
    return {
        "train_loss_report": train_loss_report,
        "val_loss_report": val_loss_report,
        "val_train_loss_report": val_train_loss_report
    }
