class Database:
    """Save the event stream inside a database to easily query it later"""

    def __init__(
        self, gv, rundir=None, runname=None, uri="sqlite:///sqlite.db"
    ) -> None:
        _, self.name = get_rundir(rundir, runname)
        self.exec = None
        self.pack = None
        self.pack_to_idx = dict()
        self.repeat = -1

        self.engine = sqlalchemy.create_engine(
            uri,
            echo=False,
            future=True,
            json_serializer=to_json,
            json_deserializer=from_json,
        )
        try:
            Base.metadata.create_all(self.engine)
        except DBAPIError:
            pass

        gv.subscribe(self.on_event)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        pass

    def create_run(self, data):
        name = data.get("#run-name")
        metadata = data.get("#meta")

        with Session(self.engine) as sess:
            self.exec = Exec(name=str(name), meta=metadata)

            sess.add(self.exec)
            sess.commit()
            sess.refresh(self.exec)

    def create_pack(self, run, data={}):
        pid = id(run)
        if self.pack is not None and pid not in self.pack_to_idx:
            self.pack_to_idx[pid] = len(self.pack_to_idx)

        elif self.pack is None:
            with Session(self.engine) as sess:
                self.pack = Pack(
                    exec_id=self.exec._id,
                    config=run,
                    name=run["name"],
                )
                self.pack_to_idx[pid] = 0

                sess.add(self.pack)
                sess.commit()
                sess.refresh(self.pack)

        idx = self.pack_to_idx[pid]
        return self.pack, idx

    def on_event(self, data):
        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)

        if "#run-start" in data:
            return self.on_exec(data)

        if "#run-end" in data:
            self.exec = None
            return

        if "#start" in data:
            run, idx = self.create_pack(run, data)
            self.save_metric("#start", data["#start"], idx)
            return

        if "#end" in data:
            run, idx = self.create_pack(run)
            self.save_metric("#end", data["#end"], idx)
            self.pack = None
            return

        if "#repeat" in data:
            self.repeat = data.get("#repeat")
            return

        if "#config" in data:
            return

        if "#stderr" in data:
            return

        if "progress" in data:
            return

        if "gpudata" in data:
            self.on_gpudata(run, data)
            return

        self.on_metric(run, data)

    def on_exec(self, data):
        self.create_run(data)

    def on_start(self, run, data):
        self.create_pack(run, data)

    def on_metric(self, run, data):
        run, idx = self.create_pack(run)

        with Session(self.engine) as sess:
            for k, v in data.items():
                self._save_metric(sess, k, v, idx)
            sess.commit()

    def on_gpudata(self, run, data):
        run, idx = self.create_pack(run)
        gpudata = data.get("gpudata", {})

        with Session(self.engine) as sess:
            for gpid, gpu in gpudata.items():
                for metric, value in gpu.items():
                    self._save_metric(
                        sess,
                        metric,
                        value,
                        idx,
                        gpid,
                    )
            sess.commit()

    def save_metric(self, key, value, idx, gpuid=-1):
        with Session(self.engine) as sess:
            self._save_metric(sess, key, value, idx, gpuid)
            sess.commit()

    def _save_metric(self, sess, key, value, idx, gpuid=-1):
        sess.add(
            Metric(
                exec_id=self.exec._id,
                pack_id=self.pack._id,
                name=key,
                value=value,
                index=idx,
                gpuid=gpuid,
                repeat=self.repeat,
            )
        )
