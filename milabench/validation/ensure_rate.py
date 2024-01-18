from .validation import BenchLogEntry, ValidationLayer


class Layer(ValidationLayer):
    """Makes sure the training rate is generated for each benchmarks"""

    def __init__(self, **kwargs) -> None:
        self.rates = dict()
        self.errors = 0

    def on_start(self, entry):
        tag = entry.tag

        if tag not in self.rates:
            self.rates[tag] = 0

    def on_data(self, entry: BenchLogEntry):
        tag = entry.tag

        if entry.data:
            self.rates[tag] += entry.data.get("rate", 0)

    def on_end(self, entry):
        self.errors += self.rates[entry.tag] <= 0

    def report(self, summary, short=True, **kwargs):
        for tag, rate in self.rates.items():
            if rate > 0:
                continue

            with summary.section(tag):
                summary.add("* no training rate retrieved")

        self.set_error_code(self.errors)
        return self.errors
