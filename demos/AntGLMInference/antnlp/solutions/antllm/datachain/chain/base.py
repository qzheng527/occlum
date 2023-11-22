#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

__author__ = "kexi"

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Union
import multiprocessing
from tqdm import tqdm


class DataChain(ABC):
    """The base class of data chain. Different from Chain in langchain, DataChain uses simpler API design.
    """

    def __init__(
        self,
        output_key: str = "result",
        max_workers: int = 20,
        remove_empty_output: bool = True,
        verbose: bool = False,
    ) -> None:
        self.output_key = output_key
        self._max_workers = max_workers
        self._remove_empty_output = remove_empty_output
        self._verbose = verbose
        self._inputs = []
        self._outputs = []

    @classmethod
    def from_config(cls, config: Union[Dict, str, Path]):
        """Build datachain from configuration

        Args:
            config (Union[Dict, str, Path]): the configuration of datachain
        """
        pass

    def batch_run(
        self,
        input_list: List[Dict[str, Any]] = None,
        concurrency: str = 'thread'
    ) -> List[Dict[str, Any]]:
        """batch version of run

        Args:
            input_list (List[Dict[str, Any]]): the batch input list
            concurrency: `str`, batch run concurrency mode, default is `thread`
                `thread` - use multi thread
                `process` - using multi process

        Returns:
            List[Dict[str, Any]]: batch results
        """
        _outputs_contain_list = False
        if not input_list:
            input_list = self._inputs

        if concurrency == 'thread':
            futures = []
            with tqdm(total=len(input_list), disable=not self._verbose) as pbar:
                with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                    for input in input_list:
                        future = executor.submit(self.run, input)
                        futures.append(future)
                self._outputs = [{} for _ in range(len(input_list))]

                for future in as_completed(futures):
                    idx = futures.index(future)
                    output = future.result()
                    if isinstance(output, list):
                        _outputs_contain_list = True
                    self._outputs[idx] = output
                    pbar.update(1)

        elif concurrency == 'process':
            num_proc = min(self._max_workers, multiprocessing.cpu_count())
            with multiprocessing.Pool(processes=num_proc) as pool:
                data_it = pool.imap(self.run, input_list, chunksize=128)
                for output in tqdm(data_it, total=len(input_list)):
                    if isinstance(output, list):
                        _outputs_contain_list = True
                    self._outputs.append(output)
        else:
            raise ValueError(f'not support concurrency mode: {concurrency}')

        # exclude empty output
        if self._remove_empty_output:
            self._outputs = [item for item in self._outputs if item]

        # check if item of outputs is list
        if _outputs_contain_list:
            _outputs = []
            for item in self._outputs:
                if isinstance(item, list):
                    _outputs.extend(item)
                    continue
                _outputs.append(item)

            self._outputs = _outputs

        return self._outputs

    @abstractmethod
    def run(
        self,
        inputs: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """process the inputs with the datachain. Note that this function doesn't touch IO processing,
        it focuses on the processing of single data item.

        Args:
            inputs (Dict[str, Any]): the input to the chain

        Returns:
            Dict[str, Any]: the result of the datachain
        """
        pass

    def save(self, output_path=None, **kwargs):
        """_summary_

        Args:
            output_path (str): the output path of the results
        """
        pass

    def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
        """_summary_

        Args:
            input_path (str): the input path of data to process
        """
        pass
