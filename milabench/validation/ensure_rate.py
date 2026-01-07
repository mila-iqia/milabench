from .validation import BenchLogEntry, ValidationLayer, group_by_benchname


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
        grouped = group_by_benchname(self.rates)

        with summary.section("Metric Collection"):
            for bench, rates in grouped.items():
                with summary.section(bench):
                    low_rates = []
                    no_rates = []

                    for tag, rate in rates:
                        if rate == 0:
                            no_rates.append(tag)
                        elif rate < 30:
                            low_rates.append(tag)

                    if len(low_rates) > 0:
                        summary.add(f"* {len(low_rates)} x Few training rate retrieved ({', '.join(low_rates)})")

                    if len(no_rates) > 0:
                        summary.add(f"* {len(no_rates)} x no training rate retrieved ({', '.join(no_rates)})")
                    
        self.set_error_code(self.errors)
        return self.errors
