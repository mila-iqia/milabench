import os
import yaml

from milabench.config import build_config


this = os.path.dirname(__file__)
config = os.path.join(this, "..", "..", "config")


def list_missing_batch_resizer():
    standard = os.path.join(config, "standard.yaml")
    scaling = os.path.join(config, "scaling.yaml")

    base_conf = build_config(standard)

    with open(scaling, "r") as fp:
        scaling = yaml.safe_load(fp)

    missing_benches = []
    def add_bench(k, tags):
        print(k, tags)
        missing_benches.append(k)

    for k, v in base_conf.items():
        if k[0] == "_":
            continue

        if not v.get("enabled", False):
            continue 

        tags = set(v.get("tags", []))

        if "nobatch" in tags:
            continue

        if k in scaling:
            s = scaling[k].get("model", {})

            if len(s) <= 1:
                add_bench(k, tags)
        else:
            add_bench(k, tags)



    b = [f"\"{b}\"" for b in missing_benches]

    


    print(" ".join(b))


if __name__ == "__main__":
    list_missing_batch_resizer()
