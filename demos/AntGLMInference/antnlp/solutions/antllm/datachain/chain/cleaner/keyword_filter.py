'''基于关键词过滤样本 Chain.'''
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Iterator
from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.utils import load_jsonl, dump_jsonl


class KeywordsFilterChain(DataChain):
    '''关键词过滤处理器.'''

    def __init__(
        self,
        keys: List[str],
        uncased: bool = True,
        keyword_source: Optional[List[str]] = None,
        verbose: bool = True,
        max_workers: int = 16,
    ):
        '''关键词过滤处理器, 如果命中了关键词返回 None.

        Params:
            keys: `List[str]`, 输入数据 dict 中需要关键词检查的字段, 任何一个字段命中了关键词都会过滤该样本

            uncased: `bool`, 是否匹配大小写, 默认不关心大小写

            keyword_source: `List[str]`, 过滤关键词文件来源, 每行格式为 `关键词`
                默认使用该目录下 `resources/filters` 文件夹下所有文件
        '''
        super().__init__(verbose=verbose, max_workers=max_workers)
        if not keyword_source:
            keyword_source = [Path(__file__).parent.joinpath('resources/filters')]
        self.keyword_patterns = self._load_keyword_patterns(
            keyword_source=keyword_source,
            uncased=uncased,
        )
        self.keys = keys
        self.uncased = uncased

    def save(self, output_path=None, **kwargs):
        dump_jsonl(self._outputs, output_path)

    def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
        return load_jsonl(input_path)

    def run(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        hit = False
        check_inputs = [inputs[k] for k in self.keys]
        for check in check_inputs:
            hit = self._keyword_match(check)
            if hit:
                break

        if hit:
            return None

        return inputs

    def _load_keyword_patterns(
        self,
        keyword_source: List[str],
        uncased: bool = True,
    ) -> List[re.Pattern]:
        '''关键词查询 pattern.'''
        keywords = []
        kws_files = []

        for file in keyword_source:
            file = Path(file)
            if file.is_dir():
                kws_files.extend([_ for _ in file.iterdir()])
                continue
            kws_files.append(file)

        for file in kws_files:
            if not file.exists():
                continue
            with open(file, 'r', encoding='utf-8') as fi:
                for line in fi:
                    keywords.append(line.strip())

        if uncased:
            keywords = [item.lower() for item in keywords]

        patterns = [re.compile(item) for item in keywords]
        return patterns

    def _keyword_match(self, input: Union[dict, list, str]) -> bool:
        '''关键词遍历匹配.'''
        match = False

        if isinstance(input, list) or isinstance(input, Iterator):
            for item in input:
                match = self._keyword_match(item)
                if match:
                    break

        if isinstance(input, dict):
            for _, item in input.items():
                match = self._keyword_match(item)
                if match:
                    break

        if isinstance(input, str):
            for pattern in self.keyword_patterns:
                if self.uncased:
                    input = input.lower()

                if pattern.search(input):
                    match = True

        return match
