from .base import BaseEvaluator, TestTimeAugEvaluator, PoolSearchEvaluator

def get_evaluator(evaluator_type: str, filename: str) -> BaseEvaluator:

    match evaluator_type:
        case 'TTA':
            return TestTimeAugEvaluator(filename)
        case 'NoTTA':
            return PoolSearchEvaluator(filename)
        case _:
            raise ValueError(evaluator_type)