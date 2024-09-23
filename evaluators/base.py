import os

from .report import Report

class BaseEvaluator:

    def __init__(self, filename: str):
        r'''
        '''
        self.data = []
        self.filename = filename
        
    def load(self) -> None:
        if os.path.exists(self.filename):
            self.data = self.report.load(self.filename)

    def __str__(self) -> str:
        return str(self.report)

    def log(self) -> None:
        r'''Print the evaluation results and write them to `self.filename` in tsv
        '''
        print(self.report.stringfy(self.data))
        print(f'[*] Saving results to {self.filename}')
        self.report.save(self.data, self.filename)

    def update(self) -> None:
        raise NotImplementedError

class TestTimeAugEvaluator(BaseEvaluator):

    def __init__(self, filename: str):
        super().__init__(filename)
        self.report = Report(
            headers=['Model', 'Policy', 'Budget', 'TTA', 'Top-1'],
            format=[str, str, int, str, float])
        
    def update(self, 
        values: dict, 
    ) -> None:
        r'''Update the accuracy meters
        '''
        self.data.append([
            values['model'],
            values['policy'],
            values['budget'],
            values['agg'],
            f'{values["top1"]:.2f}', 
        ])
        
class PoolSearchEvaluator(BaseEvaluator):

    def __init__(self, filename: str):
        super().__init__(filename)
        self.report = Report(
            headers=['Model', 'B', 'Criterion', 'Aggregate', 'Acc', 'Err'],
            format=[str, int, str, str, float, float])
        
    def update(self, 
        values: dict, 
    ) -> None:
        r'''Update the accuracy meters
        '''
        self.data.append([
            values['model'],
            values['budget'],
            values['criterion'],
            values['aggregate'],
            f'{values["top1_acc"]:.2f}',
            f'{values["top1_err"]:.2f}', 
        ])
    