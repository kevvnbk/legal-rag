from legalbench import evaluation
from legalbench.tasks import *

def evaluate(task, predictions, answers):
    """
    Evaluate model performance using the LegalBench evaluation script.

    :param task: Task name.
    :param predictions: Model predictions.
    :param answers: Ground truth answers.
    """
    
    return evaluation.evaluate(task, predictions, answers)