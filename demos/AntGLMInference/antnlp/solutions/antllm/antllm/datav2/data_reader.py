#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

import logging
import os
from pathlib import Path
from typing import Callable, Union

import tqdm
from multiprocess import Process, Queue

COMPLETE = "COMPLETE"
STOP = "STOP"

logger = logging.getLogger(__name__)


def default_workder(input_queue: Queue, output_queue: Queue):
    for row in iter(input_queue.get, STOP):
        if row:
            output_queue.put(row)

    output_queue.put(COMPLETE)


class DataReader:
    TASK_QUEUE_LIMIT = 10000
    DONE_QUEUE_LIMIT = 10000

    """Process large file with multi processing

    Args:
        path (str or Path): the path of the data file.
        worker_ftn (Callable): the function called by each process.
    """

    def __init__(self,
                 path: Union[str, Path],
                 worker_ftn: Callable,
                 postprocess_ftn: Callable,
                 num_processes: int = 10):
        self.path = path
        self.num_processes = num_processes
        self.worker_ftn = worker_ftn  # each process is a worker to process data concurrently

        # may not be thread safe, and will process the done queue, e.g. save to disk
        self.postprocess_ftn = postprocess_ftn

        self._paths = []
        self._process_workers = []
        self.task_queue, self.done_queue = None, None

    def process(self):
        if os.path.isdir(self.path):
            self._paths = [os.path.join(top, name) for top, _,
                           names in os.walk(self.path) for name in names]
        else:
            self._paths = [self.path]

        self.task_queue, self.done_queue = Queue(maxsize=self.TASK_QUEUE_LIMIT), Queue(
            maxsize=self.DONE_QUEUE_LIMIT)

        # create multiple processes to process data from task queue and put results to done queue
        self._process_workers = []
        for _ in range(self.num_processes):
            process = Process(target=self.worker_ftn, args=(
                self.task_queue, self.done_queue))
            process.start()
            self._process_workers.append(process)

        process = Process(target=self.read_input_to_queue)
        process.start()
        count = len(self._process_workers)
        progress_bar = tqdm.tqdm()
        while True:
            data = self.done_queue.get()
            if data == COMPLETE:
                count -= 1
                if count == 0:
                    break
            else:
                self.postprocess_ftn(data)
                progress_bar.update()
        progress_bar.close()

    def read_input_to_queue(self):
        for path in self._paths:
            logger.info(f"Start reading {path}")
            with open(path, "r", encoding="utf-8") as file:
                for row in file:
                    self.task_queue.put(row)

        logger.info("Read input complete")
        for _ in range(len(self._process_workers)):
            self.task_queue.put(STOP)
