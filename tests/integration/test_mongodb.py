from milabench.utils import multilogger
from milabench.testing import replay_run
from milabench.metrics.mongodb import MongoDB

import pandas as pd
import pymongo
from bson.json_util import dumps as to_json


TEST_INSTANCE = "mongodb://localhost:27017/"


def test_mongodb_real(runs_folder):
    # Test locally with
    # rm -rf .mongo && mkdir .mongo && mongod --dbpath .mongo/
    metric_saver = MongoDB(TEST_INSTANCE)

    with multilogger(metric_saver) as log:
        for msg in replay_run(runs_folder / "sedumoje.2023-03-24_13:57:35.089747"):
            log(msg)

    client = metric_saver.client
    db = client["milabench"]

    # This test works even if the tests ran more than once
    # and the database instance was not cleaned
    run_count = len(list(db.run.find({})))
    pack_count = len(list(db.pack.find({}))) // run_count
    metric_count = len(list(db.metrics.find({}))) // run_count

    assert pack_count == 2
    assert metric_count == 42


def test_reporting(runs_folder):
    client = pymongo.MongoClient(TEST_INSTANCE)
    db = client["milabench"]

    run = db.run.find_one({})

    agg = db.run.aggregate(
        [
            {
                "$match": {
                    "_id": run["_id"],
                },
            },
            {
                "$lookup": {
                    "from": "pack",
                    "localField": "_id",
                    "foreignField": "exec_id",
                    "as": "packs",
                },
            },
            {
                "$unwind": {"path": "$packs", "includeArrayIndex": "idx"},
            },
            {
                "$lookup": {
                    "from": "metrics",
                    "localField": "packs._id",
                    "foreignField": "pack_id",
                    "as": "metrics",
                },
            },
            {
                "$unwind": {"path": "$metrics", "includeArrayIndex": "midx"},
            },
            {
                "$project": {
                    "_id": 0,
                    "name": 1,
                    "bench": "$packs.name",
                    "tag": "$packs.tag",
                    "metric": "$metrics.name",
                    "value": {
                        "$avg": 
                            "$metrics.value"
                        ,
                    },
                }
            },
            {
                "$group": {
                    "_id":{"$concat": ["$bench", ".", "$metric"]},
                    "name": {"$first": "$name"},
                    "bench": {"$first": "$bench"},
                    "metric": {"$first": "$metric"},
                    "count": {"$count": {}},
                    "value": {
                        "$avg": "$value"
                    },
                    "sds": {
                        "$stdDevSamp": "$value"
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
    )


    print()
    print(run)

    # packs = db.pack.find(filter={"run_id": run['_id']})
    # data = db.metrics.find(filter={'pack_id': {"$in": list(p['_id'] for p in packs)}})
    # print(list(data))

    d = list(agg)
    print(to_json(d, indent=2))
    print(len(d))
    print(pd.DataFrame(d))

