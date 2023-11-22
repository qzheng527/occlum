'''关键词替换 Chain.'''
import re
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Iterator
from solutions.antllm.datachain.chain.base import DataChain
from solutions.antllm.datachain.utils import load_jsonl, dump_jsonl


class KeywordsReplaceChain(DataChain):
    '''关键词替换处理器.'''

    def __init__(
        self,
        keys: List[str],
        uncased: bool = True,
        keyword_source: Optional[List[str]] = None,
        verbose: bool = True,
        max_workers: int = 16,
    ):
        '''关键词替换处理器, 如果命中了关键词则进行替换.

        Params:
            keys: `List[str]`, 输入数据 dict 中需要关键词检查的字段, 命中了关键词都会替换

            uncased: `bool`, 是否匹配大小写, 默认不关心大小写

            keyword_source: `List[str]`, 过滤关键词文件来源, 文件格式为 jsonl: {"from": "xx", "to", "xx"}
                默认使用该目录下 `resources/replaces` 文件夹下所有文件
        '''
        super().__init__(verbose=verbose, max_workers=max_workers)
        if not keyword_source:
            keyword_source = [Path(__file__).parent.joinpath('resources/replaces')]
        self.replace_dict = self._load_replace_dict(
            keyword_source=keyword_source,
            uncased=uncased,
        )
        self.replace_pattern = re.compile(
            '|'.join(re.escape(key) for key in self.replace_dict.keys())
        )
        self.keys = keys
        self.uncased = uncased

    def save(self, output_path=None, **kwargs):
        dump_jsonl(self._outputs, output_path)

    def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
        return load_jsonl(input_path)

    def run(self, inputs: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        replaces = {k: inputs[k] for k in self.keys}
        replaces = self._keyword_replace(replaces)
        inputs.update(replaces)

        return inputs

    def _load_replace_dict(
        self,
        keyword_source: List[str],
        uncased: bool = True,
    ) -> Dict[str, str]:
        '''替换词 -> 被替换词词典.'''
        kws_files = []
        kws_dict = {}

        for file in keyword_source:
            file = Path(file)
            if file.is_dir():
                kws_files.extend([_ for _ in file.iterdir()])
                continue
            kws_files.append(file)

        for file in kws_files:
            if not file.exists():
                continue
            json_lines = load_jsonl(file)
            for item in json_lines:
                if uncased:
                    kws_dict[item['from'].lower()] = item['to'].lower()
                else:
                    kws_dict[item['from']] = item['to']

        return kws_dict

    def _keyword_replace(
        self,
        input: Union[dict, list, str],
    ) -> bool:
        '''关键词遍历替换.'''
        if isinstance(input, list) or isinstance(input, Iterator):
            for idx, item in enumerate(input):
                if isinstance(item, str):
                    input[idx] = self.replace_pattern.sub(
                        lambda x: self.replace_dict[x.group()],
                        item,
                    )
                else:
                    self._keyword_replace(item)

        if isinstance(input, dict):
            for k, item in input.items():
                if isinstance(item, str):
                    input[k] = self.replace_pattern.sub(
                        lambda x: self.replace_dict[x.group()],
                        item,
                    )
                else:
                    self._keyword_replace(item)

        if isinstance(input, str):
            input = self.replace_pattern.sub(
                lambda x: self.replace_dict[x.group()],
                input,
            )

        return input
