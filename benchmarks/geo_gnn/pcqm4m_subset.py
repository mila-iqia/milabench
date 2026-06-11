import os
import os.path as osp
import shutil
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, download_url, extract_tar
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import PCQM4Mv2
from torch_geometric.utils import from_smiles as _from_smiles
from benchmate.progress import tqdm


class PCQM4Mv2Subset(PCQM4Mv2):
    suppl_url = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz"

    def __init__(
        self,
        size: int,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        backend: str = "sqlite",
        from_smiles: Optional[Callable] = None
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
            "z": dict(dtype=torch.long, size=(-1,)),
        }

        self.from_smiles = from_smiles or _from_smiles

        self._rebuild_if_too_small(root)

        super(PCQM4Mv2, self).__init__(root, transform, backend=backend, schema=schema)

        split_idx = torch.load(self.raw_paths[1], weights_only=False)
        self._indices = split_idx[self.split_mapping[split]].tolist()

        bad_indices = self._load_bad_indices()
        if bad_indices:
            self._indices = [i for i in self._indices if i not in bad_indices]

    def _rebuild_if_too_small(self, root: str) -> None:
        """Delete processed data if the existing DB has fewer entries than requested."""
        processed_dir = osp.join(root, "processed")
        size_file = osp.join(processed_dir, "dataset_size.txt")

        if not osp.exists(processed_dir):
            return

        if not osp.exists(size_file):
            return

        with open(size_file) as f:
            existing_size = int(f.read().strip())

        if existing_size < self.size:
            print(
                f"Existing DB has {existing_size} entries but {self.size} requested. "
                f"Rebuilding..."
            )
            shutil.rmtree(processed_dir)

    @property
    def _bad_indices_path(self) -> str:
        return osp.join(self.processed_dir, "bad_indices.pt")

    def _load_bad_indices(self) -> set:
        if osp.exists(self._bad_indices_path):
            return set(torch.load(self._bad_indices_path, weights_only=False))
        return set()

    def _save_bad_indices(self, bad_indices: set) -> None:
        torch.save(sorted(bad_indices), self._bad_indices_path)

    def verify(self) -> List[int]:
        """Verify all DB entries are retrievable and valid.

        Returns a list of indices that failed verification.
        """
        bad_indices = []
        n = super(PCQM4Mv2Subset, self).__len__()
        num_entries = min(n, self.size)

        print(f"Verifying {num_entries} database entries...")
        for i in tqdm(range(num_entries), total=num_entries):
            try:
                data = self.get(i)
                if data is None:
                    bad_indices.append(i)
                    continue
                if not hasattr(data, 'edge_index') or data.edge_index is None:
                    bad_indices.append(i)
                    continue
                if not hasattr(data, 'x') or data.x is None:
                    bad_indices.append(i)
                    continue
            except (TypeError, IndexError, RuntimeError) as e:
                print(f"  Bad entry at index {i}: {e}")
                bad_indices.append(i)

        if bad_indices:
            print(f"Found {len(bad_indices)} bad entries: {bad_indices}")
            self._save_bad_indices(set(bad_indices))
            self._indices = [i for i in self._indices if i not in set(bad_indices)]
        else:
            print("All entries verified successfully.")

        return bad_indices

    def raw_file_names(self):
        return super().raw_file_names + [
            osp.join("pcqm4m-v2", "raw", "pcqm4m-v2-train.sdf")
        ]

    def download(self):
        if all(os.path.exists(path) for path in self.raw_paths):
            return

        # Download 2d graphs
        super().download()

        # Download 3D coordinates
        file_path = download_url(self.suppl_url, self.raw_dir)
        # md5sum: fd72bce606e7ddf36c2a832badeec6ab
        extract_tar(file_path, osp.join(self.raw_dir, "pcqm4m-v2", "raw"), mode="r:gz")
        os.unlink(file_path)

    def _validate_data(self, data: Data, index: int) -> bool:
        """Check that a processed data sample has all required fields."""
        if data is None:
            print(f"  Validation failed at source index {index}: data is None")
            return False
        if not hasattr(data, 'x') or data.x is None:
            print(f"  Validation failed at source index {index}: missing x")
            return False
        if not hasattr(data, 'edge_index') or data.edge_index is None:
            print(f"  Validation failed at source index {index}: missing edge_index")
            return False
        if not hasattr(data, 'pos') or data.pos is None:
            print(f"  Validation failed at source index {index}: missing pos")
            return False
        if not hasattr(data, 'z') or data.z is None:
            print(f"  Validation failed at source index {index}: missing z")
            return False
        if data.x.shape[0] == 0:
            print(f"  Validation failed at source index {index}: empty x tensor")
            return False
        return True

    def process(self) -> None:
        import pandas as pd

        df = pd.read_csv(self.raw_paths[0])

        data_list: List[Data] = []
        suppl = Chem.SDMolSupplier(self.raw_paths[-1])
        iterator = enumerate(zip(df["smiles"], df["homolumogap"], suppl))
        k = 0
        skipped = 0

        for i, (smiles, y, extra) in tqdm(iterator, total=min(len(df), self.size)):

            if extra is None:
                print(f"Skipping {i}: SDF entry is None")
                skipped += 1
                continue

            try:
                data = self.from_smiles(Chem.MolToSmiles(extra))
                data.y = y
                data.pos = torch.tensor(
                    extra.GetConformer().GetPositions(), dtype=torch.float
                )
                data.z = torch.tensor(
                    [atom.GetAtomicNum() for atom in extra.GetAtoms()], dtype=torch.long
                )
            except Exception as e:
                print(f"Skipping {i}: failed to process molecule: {e}")
                skipped += 1
                continue

            if not self._validate_data(data, i):
                skipped += 1
                continue

            k += 1
            data_list.append(data)
            if (
                k + 1 == len(df) or (k + 1) % 1000 == 0 or k >= self.size
            ):  # Write batch-wise:
                self.extend(data_list)
                data_list = []

            if k >= self.size:
                break

        if skipped > 0:
            print(f"Processing complete: {k} valid entries written, {skipped} entries skipped")

        size_file = osp.join(self.processed_dir, "dataset_size.txt")
        with open(size_file, "w") as f:
            f.write(str(k))

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
        rval["z"] = data.z
        return rval
