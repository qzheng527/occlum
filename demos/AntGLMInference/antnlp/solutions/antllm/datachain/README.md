# DataChain
DataChain是大模型相关数据处理工作的脚本库。

## How-TO
场景特定的数据处理都以Chain的形式放在`chain`文件夹下，每个Chain中有五个核心的方法：
- from_config
  该函数用于从一个config文件中加载一个Chain，用于复杂场景下的Chain管理，可以不实现。
- run
  输入输出都是一个Dict，用于单个数据项的处理，run函数不会涉及IO相关操作。
- batch_run
  batch版本的run，输入是Dict的List，输出也是List。如果实现了`load`函数，即从外部介质加载了待处理的数据，则运行batch_run的时候可以不用给输入。
- load
  从文件、ODPS等介质中读取数据，数据读取为Dict的List，存放在chain中，可以被batch_run直接消费。
- save
  将batch_run之后的处理结果保存到文件、ODPS等。

## 示例

### ChatGPT调用统一
DataChain中所有大模型的调用都统一到`AntLLMChain`，`AntLLMChain`中包括LLM调用，Prompt以及输出的解析，具体见[AntLLMChain](chain/llm/base.py). 其中：
- LLM内置了蚂蚁内部的OpenAI调用，具体见[AntOpenAI](llms/ant_openai.py)。
- Prompt传入可以是一个文件路径，也可以是一个文件名，如果是一个文件名，默认到`prompts/resources`下搜索。
- OutputParser统一放在`output_parsers`文件夹下，用户根据自己的场景自由制定需要如何Parse LLM的输出。

```python
llm = AntOpenAI()
llm_chain = AntLLMChain(llm, "your prompt name")
res = llm_chain.run({"key1": "val1"})
```

### 去毒数据回收
以去毒场景中一个简单的数据回收流程为例，去毒场景需要从ODPS中回收数据，并且将数据中的部分字段使用ChatGPT改写，然后将数据再按照某种格式保存。

数据读取
```python
def load(self, input_path=None, **kwargs) -> List[Dict[str, Any]]:
    odps_project_table = kwargs.get("odps_project_table", input_path)
    columns = kwargs.get("columns", None)
    access_id = kwargs.get("access_id", None)
    access_key = kwargs.get("access_key", None)
    project = kwargs.get("project", None)
    endpoint = kwargs.get("endpoint", None)
    odps = ODPSReader(odps_project_table, columns, access_id, access_key, project, endpoint)
    self._inputs = odps.read()
```

数据处理
```python
def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    prompt = inputs[self.prompt_column]
    candidates = inputs[self.candidates_column]
    gold = inputs[self.gold_column]

    rewrite_prompt = self.llm.run({PROMPT: prompt})
    prompts = [prompt]
    if OUTPUT in rewrite_prompt and rewrite_prompt[OUTPUT] != FAILED:
        if isinstance(rewrite_prompt[OUTPUT], List):
            prompts.extend(rewrite_prompt[OUTPUT])
        else:
            prompts.append(rewrite_prompt[OUTPUT])

    return {"prompts": prompts, "candidates": candidates, "gold": gold}
```

保存
```python
def save(self, output_path=None, **kwargs):
    with open(output_path, "w", encoding="utf-8") as fo:
        for item in self._outputs:
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
```
