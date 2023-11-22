#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import mmap
import os
import random
from pathlib import Path
from typing import Union


def get_scatter_offset_path(path: str, scatter_idx: Union[str, int]):
    """Get the part file and offset file in a scatter directory given a scatter index, e.g. 1.part and 1.offset

    Args:
        path (str, Optional): the scatter directory
        scatter_idx (str or int, Required): the index of scatter file.
    """
    return os.path.join(path, str(scatter_idx) + ".part"), os.path.join(path, str(scatter_idx) + ".offset")


def check_scatter_path(path, scatter_num, has_offset):
    """Check whether a scatter directory is valid.

    Args:
        path (str): the scatter directory
        scatter_num (int): the expected number of scatter files in the directory
        has_offset (bool): if True, offset file is also expected.
    """
    for i in range(scatter_num):
        part_path, offset_path = get_scatter_offset_path(path, i)
        if not os.path.exists(part_path):
            return False
        if has_offset and not os.path.exists(offset_path):
            return False

    return True


def exist_scatter_directory(path):
    return os.path.exists(get_scatter_directory(path))


def get_scatter_directory(path: str):
    """Get the scatter directory given a data path.
    If the path is a directory and ends with .scatter, then we treat it as a scatter directory.
    Otherwise, we use the path name with .scatter suffix as the scatter directory.

    Args:
        path (str): the data path
    """
    if os.path.isdir(path) and path.rstrip("/").rstrip("\\").endswith(".scatter"):
        return path
    return str(Path(path).parent / Path(path).name) + ".scatter"


class ScatterWriter:
    """Write data to scatter files

    Args:
        path (str): the path to write the data
        scatter_num (int): the number of files to write the data. Default: 8
    """

    def __init__(self, path, scatter_num=8):
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(
                "the path should be directory but a file is found.")
        if not os.path.exists(path):
            Path(path).mkdir(parents=True, exist_ok=True)

        self.scatter_num = scatter_num
        self.data_paths = []
        self.data_writers = []

        self.offset_paths = []
        self.offset_writers = []

        self.offsets = []

        assert scatter_num > 0, f"scatter_num must be larger than 0, but set as {scatter_num}"

        for i in range(scatter_num):
            data_path, offset_path = get_scatter_offset_path(path, i)
            self.data_paths.append(data_path)
            self.data_writers.append(open(data_path, "w", encoding="utf-8"))
            self.offset_paths.append(offset_path)
            self.offset_writers.append(
                open(offset_path, "w", encoding="utf-8"))
            self.offsets.append(0)

    def write(self, sample: str):
        scatter_idx = random.choice(list(range(self.scatter_num)))
        line = sample + "\n"
        self.data_writers[scatter_idx].write(line)
        self.offset_writers[scatter_idx].write(
            f"{self.offsets[scatter_idx]},{len(sample.encode())}\n")
        self.offsets[scatter_idx] += len(line.encode())

    def close(self):
        for w in self.data_writers + self.offset_writers:
            w.close()


class LazyLoaderV2:
    """
    Load file line-by-line as array, support load file without loading it into memory.

    Args:
        path (str, Required): the data path
        load_memory(bool, Optional): if True, load the data into memory.
        offset_path (str, Optional): if load memory is true, offset path is required.
        mem_map (bool, Optional): if True, using mem map.
    """

    def __init__(self, path, load_memory=False, offset_path=None, mem_map=False):
        self.path = path
        self.load_memory = load_memory
        self.mem_map = mem_map
        self.offset = []

        if not load_memory:
            # if we don't load the data into memory, offset path is required
            assert offset_path and os.path.exists(offset_path)
            self.off_path = offset_path
            for line in open(self.off_path, "r"):
                arr = line.strip().split(",")
                self.offset.append((int(arr[0]), int(arr[1])))

        self._file = open(self.path, "r")

        self.samples = None
        if self.load_memory:
            self.samples = []
            for line in self._file:
                self.samples.append(line.rstrip())
        else:
            if self.mem_map:
                self.samples = mmap.mmap(
                    self._file.fileno(), 0, prot=mmap.PROT_READ)
            else:
                self.samples = self._file

    def __getitem__(self, index):
        """
        read file and splice strings based on string ending array `self.ends`
        """
        if self.load_memory:
            # pure text
            return self.samples[index]
        elif not self.mem_map:
            content = self.samples.seek(
                self.offset[index][0]).read(self.offset[index][1])
        else:
            content = self.samples[self.offset[index][0]: self.offset[index][0] + self.offset[index][1]]
            content = content.decode("utf-8", "strict")
        return content

    def __len__(self):
        if self.load_memory:
            return len(self.samples)
        return len(self.offset)

    def close(self):
        self._file.close()
