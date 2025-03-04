from legalbench import evaluation

def evaluate_model(task, predictions, answers):
    """
    Evaluate model performance using the LegalBench evaluation script.

    :param task: Task name.
    :param predictions: Model predictions.
    :param answers: Ground truth answers.
    """
    return evaluation.evaluate(task, predictions, answers["answer"].tolist())