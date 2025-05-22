#
#   Make a web dashboard instead of the CLI version
#   Easier to read the logs 
#   plots ?
#


# Have a client validation that push data to the server ?
# Server both reads the folder and the event

if False:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class _EventHandler(FileSystemEventHandler):
        def on_created(self, event):
            print(f"File created: {event.src_path}")

        def on_modified(self, event):
            print(f"File modified: {event.src_path}")

        def on_closed(self, event):
            print(f"File closed: {event.src_path}")


    class ResultFolderMonitor:
        def __init__(self, folder_path):
            self.folder_path = folder_path
            self.observer = Observer()
            self.observer.schedule(_EventHandler(), self.folder_path, recursive=True)
            self.observer.start()

        def stop(self):
            self.observer.stop()
            self.observer.join()



#
# Could we modify `SQLAlchemy` observer to make it send websocket events to the dashboard ?
# 


#
#   This reads an exisitng result folder and gives a view of the data
#

class ResultFolderInspector:
    """This is used to give a view of the results folder similar to the SQLAlchemy queries"""
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def ls_execs(self):
        for file in os.listdir(self.folder_path):
            yield self.make_exec(file)

    def load_meta(self, exec_path):
        for file in os.listdir(exec_path):

            if file.endswith(".data"):
                with open(os.path.join(exec_path, file), "r") as f:
                    for line in f.readlines():
                        line = json.loads(line)

                        if line["event"] == "meta":
                            return line["data"]
        return {}

    def make_exec(self, exec_name):
        exec_path = os.path.join(self.folder_path, exec_name)

        meta = self.load_meta(exec_path)

        return {
            "_id": exec_name,
            "name": exec_name,
            "namespace": "N/A",
            "created_time": os.path.getmtime(exec_path),
            "meta": meta,
            "status": "pending"   
        }
    
    def load_pack_info(self, exec_path, pack_name):
        pack_file = os.path.join(exec_path, pack_name)

        config = {}
        start = {}
        end = {}
        early_stop = False

        with open(os.path.join(exec_path, file), "r") as f:
            for line in f.readlines():
                line = json.loads(line)

                # TODO: handle early stopping too
                match line["event"]:
                    case "config":
                        config = line["data"]
                    case "start":
                        start = line["data"]
                    case "end":
                        end = line["data"]
                    case "stop":
                        early_stop = True
                    case _:
                        pass
  
        status = ""
        match (end.get("return_code"), early_stop):
            case None, _:
                status = "pending"
            case _, True:
            case 0, False:
                status = "good"
            case _:
                status = "error"

        return config, start, status

    def make_pack(self, exec_name, pack_name):
        name = pack_name.split(".")[0]

        config, start, status = self.load_pack_info(exec_path, pack_name)

        return {
            "_id": pack_name,
            "exec_id": exec_name,
            "name": name,
            "tag": config["tag"],
            "created_time": os.path.getctime(os.path.join(exec_path, pack_name)),
            "config": config,
            "command": start.get("command", []),
            "status": status
        }

    def make_observation(self, data):
        if "progress" in data:
            return []

        if "gpudata" in data:
            data = []

            for gid, gpudata in data["gpudata"].items()
                for key, value in gpudata.items():
                    data.append({
                        "name": key,
                        "value": value,
                        "gpu_id": gid,
                    })

            return data

        if "loss" in data:
            return [{"name": "loss", "value": data["loss"]}]

        if "rate" in data:
            return [{"name": "rate", "value": data["rate"], "units": data["units"]}]

        return []

    def make_metrics(self, exec_name, pack_name):
        pack_file = os.path.join(exec_path, pack_name)
        i = 0

        with open(os.path.join(exec_path, file), "r") as f:
            for line in f.readlines():
                line = json.loads(line)

                match line["event"]:
                    case "data":
                        i += 1

                        base = {
                            "_id": i,
                            "exec_id": exec_name,
                            "pack_id": pack_name,
                            "order": i,
                            "namespace": line.get("task", ""),
                            "job_id": job_id,
                            "gpu_id": gpu_id,
                        }

                        for observation in self.make_observation(line["data"]):
                            yield {**base, **observation}

    def make_stream(self, exec_name, pack_name, pipe):
        pack_file = os.path.join(exec_path, pack_name)

        with open(pack_file, "r") as f:
            for line in f.readlines():
                line = json.loads(line)

                if line["event"] == "line" and line["pipe"] == pipe:
                    yield line["data"]

    def make_stdout(self, exec_name, pack_name):
        return self.make_stream(exec_name, pack_name, "stdout")

    def make_stderr(self, exec_name, pack_name):
        return self.make_stream(exec_name, pack_name, "stderr")

    def latest_stdout(self, exec_name, pack_name, n=10):
        return deque(self.make_stdout(exec_name, pack_name), maxlen=n)

    def latest_stderr(self, exec_name, pack_name, n=10):
        return deque(self.make_stderr(exec_name, pack_name), maxlen=n)