#!/usr/bin/env python

import os
from benchmate.datagen import generate_fakeimagenet





if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")
    from dinov2.data.datasets import ImageNet

    data_directory = os.environ["MILABENCH_DIR_DATA"]
    dest = os.path.join(data_directory, f"FakeImageNet")


    # class_id, class_name
    with open(dest + "/labels.txt", "w") as fp:
        for i in range(1000):
            fp.write(f"{i}, {i}\n")

    # 
    # generate_fakeimagenet()



    for split in ImageNet.Split:
        dataset = ImageNet(split=split, root=dest, extra=dest)
        dataset.dump_extra()
