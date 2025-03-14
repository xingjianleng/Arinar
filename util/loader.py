import os
import numpy as np

import io
import json
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target


class CachedH5Folder(Dataset):
    def __init__(self, root):
        h5_file = os.path.join(root, "mar_cache.h5")
        h5_json = os.path.join(root, "mar_cache_h5.json")

        with open(h5_json, "r") as f:
            cache = json.load(f)

        label_names = sorted(list(set([elem.split("/")[0] for elem in cache])))
        self.label2idx = {label: i for i, label in enumerate(label_names)}
        self.samples = {}
        for c in cache:
            self.samples[c] = self.label2idx[c.split("/")[0]]
        self.samples = list(self.samples.items())

        self.h5_file = h5py.File(h5_file, "r")

    def __len__(self):
        return len(self.samples)

    def __del__(self):
        self.h5_file.close()

    def __getitem__(self, index):
        path, target = self.samples[index]

        data = self.h5_file[path]
        data = np.load(io.BytesIO(np.array(data)))

        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target
