from argparse import Namespace
from collections import defaultdict

from milabench.utils import multilogger
from milabench.testing import replay_run
from milabench.metrics.mongodb import MongoDB

import pymongo
from bson import ObjectId


class _MockCollection:
    def __init__(self, documents) -> None:
        self.documents = documents

    def update_one(self, filter, update):
        id = filter.get("_id")
        doc = self.documents[id]

        for op, val in update.items():
            if op == "$set":
                for k, v in val.items():
                    doc[k] = v

    def insert_one(self, document):
        t = ObjectId()
        document["_id"] = t
        self.documents[t] = document
        print(document)
        return Namespace(inserted_id=t)

    def bulk_write(self, documents):
        r = []
        for d in documents:
            r.append(self.insert_one(d._doc))
        return r


class _MockDatabase:
    def __init__(self, collections) -> None:
        self.collections = collections

    def __getitem__(self, item):
        return _MockCollection(self.collections[item])


class MongoMock:
    def __init__(self, value=None) -> None:
        self.database = defaultdict(lambda: defaultdict(dict))

    def construct(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        return _MockDatabase(self.database[item])


def test_mongodb(runs_folder, monkeypatch):
    mock = MongoMock()

    def client(*args, **kwargs):
        return mock.construct(*args, **kwargs)

    monkeypatch.setattr(pymongo, "MongoClient", client)

    with multilogger(MongoDB(None)) as log:
        for msg in replay_run(runs_folder / "sedumoje.2023-03-24_13:57:35.089747"):
            log(msg)

    db = mock.database["milabench"]

    assert len(mock.database) == 1
    assert len(db) == 3

    assert len(db["run"]) == 1
    assert len(db["pack"]) == 2
    assert len(db["metrics"]) == 42
