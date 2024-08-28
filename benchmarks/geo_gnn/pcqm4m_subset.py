import os
import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, download_url, extract_tar
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.utils import from_smiles as _from_smiles
from tqdm import tqdm


class PCQM4Mv2Subset(PCQM4Mv2):
    suppl_url = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"

    def __init__(
        self,
        size: int,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        backend: str = "sqlite",
        from_smiles: Optional[Callable] = None,
    ) -> None:
        assert split in ["train", "val", "test", "holdout"]

        self.size = size

        schema = {
            "x": dict(dtype=torch.int64, size=(-1, 9)),
            "edge_index": dict(dtype=torch.int64, size=(2, -1)),
            "edge_attr": dict(dtype=torch.int64, size=(-1, 3)),
            "smiles": str,
            "pos": dict(dtype=torch.float32, size=(-1, 3)),
            "y": float,
        }

        self.from_smiles = from_smiles or _from_smiles
        super(PCQM4Mv2, self).__init__(root, transform, backend=backend, schema=schema)

        split_idx = torch.load(self.raw_paths[1])
        self._indices = split_idx[self.split_mapping[split]].tolist()

    def raw_file_names(self):
        return super().raw_file_names + [
            osp.join("pcqm4m-v2", "raw", "pcqm4m-v2-train.sdf")
        ]

    def download(self):
        print(self.raw_paths)
        if all(os.path.exists(path) for path in self.raw_paths):
            return

        # Download 2d graphs
        print(self.raw_dir)
        super().download()

        # Download 3D coordinates
        file_path = download_url(self.suppl_url, self.raw_dir)
        # md5sum: fd72bce606e7ddf36c2a832badeec6ab
        extract_tar(file_path, osp.join(self.raw_dir, "pcqm4m-v2", "raw"), mode="r:gz")
        os.unlink(file_path)

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        data_list: List[Data] = []
        suppl = Chem.SDMolSupplier(self.raw_paths[-1])
        iterator = enumerate(zip(df["smiles"], df["homolumogap"], suppl))
        for i, (smiles, y, extra) in tqdm(iterator, total=min(len(df), self.size)):
            # data = from_smiles(smiles)
            data = self.from_smiles(Chem.MolToSmiles(extra))
            data.y = y
            data.pos = torch.tensor(
                extra.GetConformer().GetPositions(), dtype=torch.float
            )

            data_list.append(data)
            if (
                i + 1 == len(df) or (i + 1) % 1000 == 0 or i >= self.size
            ):  # Write batch-wise:
                self.extend(data_list)
                data_list = []

            if i >= self.size:
                break

    def __len__(self):
        return min(super().__len__(), self.size)

    def len(self):
        return min(super().len(), self.size)

    def mean(self):
        return np.mean([self.get(i).y for i in range(len(self))])

    def std(self):
        return np.std([self.get(i).y for i in range(len(self))])

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        rval = super().serialize(data)
        rval["pos"] = data.pos
        return rval
