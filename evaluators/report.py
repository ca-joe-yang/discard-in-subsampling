import tabulate

class Report:

    def __init__(self, 
        headers: list[str], 
        format: list,
        logfmt: str = 'simple_outline', 
        savefmt: str = 'tsv',
        logidx: int = -1 # Only show the first 5 column
    ):
        self.headers = headers
        self.format = format
        self.logfmt = logfmt
        self.savefmt = savefmt
        if logidx == -1:
            logidx = len(self.headers)
        self.logidx = logidx

    def stringfy(self, data) -> str:
        return tabulate.tabulate(
            [ d[:self.logidx] for d in data],
            headers=self.headers[:self.logidx],
            tablefmt=self.logfmt,
            floatfmt='.2f'
        )

    def save(self, data, filename: str) -> None:
        with open(filename, 'w') as f:
            f.write(tabulate.tabulate(
                data,
                headers=[],
                tablefmt=self.savefmt,
                floatfmt='.2f'
            ))

    def load(self, filename: str) -> list:
        data = []
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split()
                assert(len(tokens) == len(self.headers))
                data.append(
                    [ f(t) for t, f in zip(tokens, self.format) ]
                )
        return data