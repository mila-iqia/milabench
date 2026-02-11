from .push import push_server


__all__ = [
    "push_server"
]



class DataSource:
    """Interface to milabench metric data"""


class RunFolder(DataSource):
    def __init__(self, folder) -> None:
        self.src = folder

    def list_runs(self):
        return os.listdir(self.src)
    
    def list_benchmarks(self, run):
        # NOTE: group by device ?
        benchmarks = {}

        for file in os.listdir(os.path.join(self.src, run)):
            benchmarks.add(file.split(".")[0])

        return list(benchmarks)

    def metrics(self, run, benchmark):
        files = []

        for file in os.listdir(os.path.join(self.src, run)):
            if file.startswith(benchmark):
                files.append(file)
 
        for file in files:
            with open(file, "r") as fp:
                for line in fp.readlines():
                    yield json.load(line)


class Database(DataSource):
    def __init__(self, uri) -> None:
        pass



class Dashboard:
    def home():
        pass

    def push():
        pass

    def query():
        # Aggregate view
        # Compare view
        pass

    def view():
        pass

    def compare():
        pass

    def monitor():
        pass

    def report():
        pass
