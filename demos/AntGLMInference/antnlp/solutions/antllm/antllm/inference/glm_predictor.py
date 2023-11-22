import argparse
import json
import os

from typing import List, Tuple, Optional
from dataclasses import dataclass
import enum
import numpy as np
import torch
import string
from contextlib import contextmanager

from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset,
)
from solutions.antllm.antllm.data.dataset.glm_fot_dataset import GLMFoTDataset
from solutions.antllm.antllm.models.glm.modeling_glm import GLMForConditionalGeneration
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.utils.version_utils import is_oldest_version
from torch.nn import CrossEntropyLoss
from transformers.tokenization_utils_base import BatchEncoding


class GenFinishReason(enum.Enum):
    # 遇到结束符，正常结束
    EOS = 1
    # 触发生成的最大长度
    MAX_LEN = 2


@dataclass
class CompletionOutput:
    # 输出文本
    texts: List[str] = None
    finish_reasons: List[GenFinishReason] = None


class GLMForInference(object):
    """
    用于评测的GLM类模型
    """

    def __init__(
        self,
        path: str,
        adapter_path: str = None,
        gpu_index: int = None,
        multi_gpu: bool = False,
        torch_dtype=torch.float16,
        use_long_glm: bool = False,
        long_glm_type: str = 'fot',
        attention_grouping: Optional[Tuple[int, int]] = None,
        do_quantization: bool = False,
        quantization_type: str = "nf4",
        quantization_bit: int = 4
    ) -> None:
        # 检查path是否可正常访问
        assert os.path.exists(path), "Path '{}' does not exist.".format(path)

        # 检测模型的版本
        ret = is_oldest_version(path)
        if ret is True:
            self.is_oldest_version = True
            self.mask = "[sMASK]"
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import (
                GLMChineseTokenizer,
            )

            self.tokenizer = GLMChineseTokenizer.from_pretrained(path)
        elif ret is False:
            self.is_oldest_version = False
            self.mask = "[gMASK]"
            self.tokenizer = GLMTokenizer.from_pretrained(path)
        else:
            print("模型目录中词典文件缺失")
            self.tokenizer = None
        # 读取超参配置
        try:
            self.training_args = json.load(
                open(os.path.join(path, "hyper_parameters.json"), "r")
            )
        except Exception:
            self.training_args = {}
        print(f'max_length: {self.training_args.get("max_length", 1024)}')

        self.use_long_glm = use_long_glm
        self.long_glm_type = long_glm_type
        config = GLMConfig.from_pretrained(path)

        if use_long_glm is True:
            config.focused_attention = True
            config.use_cache = True
            config.attention_grouping = attention_grouping
            if long_glm_type != "fot":
                config.rotary_type = long_glm_type
        self.rotary_type = config.rotary_type

        dist_gpu_nums = torch.cuda.device_count()
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            if gpu_index is None:
                self.device = torch.device('cuda')
            else:
                self.device = torch.device(f'cuda:{gpu_index}')

        if not os.path.exists(os.path.join(path, 'pytorch_model.bin.index.json')):
            if multi_gpu and dist_gpu_nums > 1:
                from accelerate import dispatch_model
                device_map = self.auto_configure_device_map(
                    dist_gpu_nums, False)
                if do_quantization:
                    from transformers import BitsAndBytesConfig

                    self.model = GLMForConditionalGeneration.from_pretrained(
                        path, config=config, torch_dtype=torch_dtype,
                        load_in_4bit=quantization_bit == 4,
                        load_in_8bit=quantization_bit == 8,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=quantization_bit == 4,
                            load_in_8bit=quantization_bit == 8,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=torch_dtype,
                            bnb_4bit_use_double_quant=quantization_type == "nf4",
                            bnb_4bit_quant_type=quantization_type,
                        )
                    )
                else:
                    self.model = GLMForConditionalGeneration.from_pretrained(
                        path, config=config)

                self.model = dispatch_model(self.model, device_map=device_map)
                print(
                    f'model deployed on {dist_gpu_nums} gpus with dtype {torch_dtype}')
            else:
                if not torch.cuda.is_available():
                    # 跑 ACI 需要
                    print('gpu not found, use cpu')
                    device_map = {'': 'cpu'}
                else:
                    if gpu_index is None:
                        device_map = {'': 0}
                    else:
                        device_map = {'': gpu_index}
                if torch.cuda.is_available():
                    init_torch_type = torch_dtype
                else:
                    init_torch_type = None

                if do_quantization:
                    from transformers import BitsAndBytesConfig

                    self.model = GLMForConditionalGeneration.from_pretrained(
                        path, config=config, device_map=device_map, torch_dtype=init_torch_type,
                        load_in_4bit=quantization_bit == 4,
                        load_in_8bit=quantization_bit == 8,
                        quantization_config=BitsAndBytesConfig(
                            load_in_4bit=quantization_bit == 4,
                            load_in_8bit=quantization_bit == 8,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False,
                            bnb_4bit_compute_dtype=init_torch_type,
                            bnb_4bit_use_double_quant=quantization_type == "nf4",
                            bnb_4bit_quant_type=quantization_type,
                        )
                    )
                else:
                    self.model = GLMForConditionalGeneration.from_pretrained(
                        path, config=config, device_map=device_map, torch_dtype=init_torch_type)
        else:
            from accelerate import load_checkpoint_and_dispatch
            from accelerate import init_empty_weights
            config = GLMConfig.from_pretrained(path)
            with init_empty_weights():
                self.model = GLMForConditionalGeneration(config)
            self.model.tie_weights()
            device_map = 'auto'
            if gpu_index is not None:
                device_map = {'': gpu_index}
            self.model = load_checkpoint_and_dispatch(
                self.model, path, device_map=device_map, no_split_module_classes=[
                    'GLMBlock']
            )

        if self.training_args.get("peft_type", False):
            from solutions.antllm.antllm.models.peft.modeling_peft import (
                PeftModel,
            )  # noqa

            # peft_config = json.load(
            #     open(os.path.join(path, "adapter_config.json"), "r")
            # )
            # peft_config["inference_mode"] = False
            # json.dump(
            #     peft_config,
            #     open(os.path.join(path, "adapter_config.json"), "w"),
            #     ensure_ascii=False,
            #     indent=2,
            # )
            self.model = PeftModel.from_pretrained(
                self.model, path, is_trainable=True, adapter_name="default")

            if do_quantization:
                for name, module in self.model.named_modules():
                    if 'norm' in name:
                        module = module.to(torch_dtype)
                    if 'word_embeddings' in name:
                        if hasattr(module, 'weight'):
                            if module.weight.dtype == torch.float32:
                                module = module.to(torch_dtype)

        multi_lora_dir = adapter_path if adapter_path is not None else os.path.join(
            path, "adapters")
        if os.path.exists(multi_lora_dir) and not os.path.isfile(multi_lora_dir):
            from solutions.antllm.antllm.models.peft.modeling_peft import (
                PeftModel,
            )  # noqa
            peft_init_flag = isinstance(self.model, PeftModel)

            adapter_names = os.listdir(multi_lora_dir)
            print(f"Load multi-adapters: {adapter_names}")
            for adapter_name in adapter_names:
                adapter_path = os.path.join(multi_lora_dir, adapter_name)
                if not os.path.isfile(adapter_path):
                    if peft_init_flag is True:
                        self.model.load_adapter(
                            adapter_path, adapter_name=adapter_name
                        )
                    else:
                        peft_init_flag = True
                        self.model = PeftModel.from_pretrained(
                            self.model, adapter_path, adapter_name=adapter_name)
            if do_quantization:
                for name, module in self.model.named_modules():
                    if 'norm' in name:
                        module = module.to(torch_dtype)
                    if 'word_embeddings' in name:
                        if hasattr(module, 'weight'):
                            if module.weight.dtype == torch.float32:
                                module = module.to(torch_dtype)

        if torch.cuda.is_available() and not do_quantization:
            self.model.to(dtype=torch_dtype)
        self.model.eval()
        # 由于训练阶段保持单次forward，inference阶段需要设置max_memory_length>0
        self.model.set_max_memory_length(
            self.training_args.get("max_length", 1024))
        self.eos_token = self.training_args.get('eos_token', '<|endofpiece|>')
        print(f'eos_token: {self.eos_token}')
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.eos_token)
        self.rotary_type = self.training_args.get('rotary_type', 'none')
        print(f'rotary_type: {self.rotary_type}')
        self.reset_session()

    def auto_configure_device_map(self, dist_gpu_nums, is_peft_model=False):
        # lora引入后模型的结构及命名会改变
        layer_prefix = 'base_model.model.glm' if is_peft_model else 'glm'
        # transformer.layers 占用 48 层；其他层默认放入cuda:0
        num_trans_layers = 48
        per_gpu_layers = 52 / dist_gpu_nums
        # 设备映射表
        device_map = {
            f'{layer_prefix}.word_embeddings': 0,
            f'{layer_prefix}.transformer.final_layernorm': 0,
            f'{layer_prefix}.transformer.position_embeddings': 0,
            f'{layer_prefix}.transformer.block_position_embeddings': 0
        }
        current_device_used = 4
        gpu_target = 0
        for i in range(num_trans_layers):
            if current_device_used >= per_gpu_layers:
                gpu_target += 1
                current_device_used = 0
            # 超过gpu总数默认放到最后一层
            if gpu_target > (dist_gpu_nums - 1):
                gpu_target = dist_gpu_nums - 1
            device_map[f'{layer_prefix}.transformer.layers.{i}'] = gpu_target
            current_device_used += 1
        return device_map

    def _post_process(
        self,
        output,
        special_tokens=[],
        max_allowed_duplicate_times=2,
        min_len_duplicate=10,
    ):
        # if self.is_oldest_version:
        #     mask = '[sMASK]'
        # else:
        #     mask = '[gMASK]'
        # sop = "<|startofpiece|>"
        # sop_pos = outputs.index(sop)
        # output = outputs[sop_pos + len(sop):]
        for token in special_tokens:
            output = output.replace(token, "")
        output = output.replace("<n>", "\n")
        return output

    def get_confidence_score(self, query, answer, language="zh") -> float:
        """
        返回问题和答案的置信度分数

        Args:
            query:问题
            answer:答案
            language:语种,目前仅支持中文和英文,默认中文,在"zh"和"en"中选择
        """
        if language == "zh":
            scheme = "问题: {} \n 回答: {} \n 这个回答是正确的吗\nA、正确\nB、不正确\n这个回答是:"
        else:
            scheme = "Question: {} \n Answer: {} \n Is the answer:\nA、True\nB、False\nThe answer is:"
        tokenizer_A_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("A")
        )[0]
        tokenizer_B_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize("B")
        )[0]
        data = {"input": scheme.format(query, answer), "output": "A"}
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        features = GLMInstructionDataset.build_feature_from_sample(
            data,
            self.tokenizer,
            max_length=self.training_args.get("max_length", 1024),
            gpt_data=self.training_args.get("gpt_model", False),
            mask_id=mask_id,
            eos_token=self.eos_token,
            old_version_tokenizer=self.is_oldest_version,
            rotary_type=self.rotary_type,
        )
        features = {
            key: torch.Tensor([features[key]]).long().to(self.device)
            for key in features
        }
        output = self.model(**features)
        label_index = (
            (np.array(features["labels"].cpu())
             != -100).tolist()[0].index(True)
        )
        A_score = output.logits[0][label_index][tokenizer_A_id]
        B_score = output.logits[0][label_index][tokenizer_B_id]
        confidence_score = torch.softmax(
            torch.cat((torch.unsqueeze(A_score, -1),
                      torch.unsqueeze(B_score, -1))), -1
        )[0].item()
        return confidence_score

    def batch_answer_with_options(
        self,
        datas: List[dict],
        batch_size=1,
        max_input_length=-1,
        max_output_length=-1,
        likelihood=False,
        option_rank: str = "loss",
        left_truncate=True,
    ) -> List[dict]:
        """
        评估函数，给定输入和options输出选项，模型会根据loss/logit进行有限集输出选择

        Args:
            datas (string): 
                需要使用的peft模块的名字，对应于模型加载目录中`adapters`目录下每个字目录的名字。
            option_rank (string):
                option排序选择方式，按照候选项的loss值或者候选项单token的logit分数

        Example::
        >>> datas = [
        >>>        {"input": "你在哪里", "options": ["我在北京", "我在学习"]},
        >>>        {"input": "月亮在哪里", "options": ["月亮绕着地球周期旋转", "在宇宙中"]}
        >>>   ]
        >>> answer = batch_answer_with_options(datas)
        >>> print(answer)
        """
        option2value_list = []
        value_list = []
        batch_features = {}
        batch_option_lengths = []
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        value_index = 0
        max_length = self.training_args.get("max_length", 1500)
        alphabet = string.ascii_uppercase
        if max_input_length != -1 and max_output_length != -1:
            data_max_length = max_input_length + max_output_length + 4
            max_length = data_max_length if data_max_length < max_length else max_length
        if option_rank not in ["loss", "logit"]:
            raise ValueError("The value of option must be loss or logit.")
        if option_rank == "loss":
            for d, data in enumerate(datas):
                option2value = {}
                for o, option in enumerate(data["options"]):
                    data["output"] = option
                    option2value[option] = value_index  # 暂存loss分数下标
                    value_index += 1
                    mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
                    features = GLMInstructionDataset.build_feature_from_sample(
                        data,
                        self.tokenizer,
                        max_length=max_length,
                        gpt_data=self.training_args.get("gpt_model", False),
                        mask_id=mask_id,
                        old_version_tokenizer=self.is_oldest_version,
                        eos_token=self.eos_token,
                        left_truncate=left_truncate,
                        rotary_type=self.rotary_type,
                    )
                    if batch_features:
                        for key, value in features.items():
                            batch_features[key].append(value)
                    else:
                        batch_features = {key: [value]
                                          for key, value in features.items()}
                    # for i in range(len(features['labels'])):
                    #     if features['labels'][i] == self.eos_token_id:
                    #         features['labels'][i] = -100
                    option_length = sum(np.array(features["labels"]) != -100)
                    batch_option_lengths.append(option_length)
                    if len(batch_option_lengths) >= batch_size or (
                        d == len(datas) - 1 and o == len(data["options"]) - 1
                    ):
                        batch_features = {
                            key: torch.Tensor(
                                batch_features[key]).long().to(self.device)
                            for key in batch_features
                        }
                        with torch.no_grad():
                            output = self.model(**batch_features)
                        loss = loss_fct(
                            output["logits"].view(-1,
                                                  output["logits"].size(-1)),
                            batch_features["labels"].view(-1),
                        )
                        loss = torch.sum(
                            loss.view(len(batch_option_lengths), -1), -1
                        ).tolist()
                        value_list.extend(
                            [x / y for x, y in zip(loss, batch_option_lengths)]
                        )
                        batch_features = {}
                        batch_option_lengths = []
                option2value_list.append(option2value)
        else:
            # 需要option值是单token
            option_token_ids = []
            for d, data in enumerate(datas):
                option_token_id = []
                option2value = {}
                for option in data["options"]:
                    option_token_id.append(self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(option))[0])
                    option2value[option] = value_index  # 暂存value分数下标
                    value_index += 1
                option_token_ids.append(option_token_id)
                data["output"] = "A"
                mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
                features = GLMInstructionDataset.build_feature_from_sample(
                    data,
                    self.tokenizer,
                    max_length=max_length,
                    gpt_data=self.training_args.get("gpt_model", False),
                    mask_id=mask_id,
                    old_version_tokenizer=self.is_oldest_version,
                    eos_token=self.eos_token,
                    left_truncate=left_truncate,
                    rotary_type=self.rotary_type,
                )
                if batch_features:
                    for key, value in features.items():
                        batch_features[key].append(value)
                else:
                    batch_features = {key: [value]
                                      for key, value in features.items()}

                if len(list(batch_features.values())[0]) >= batch_size or (
                    d == len(datas) - 1 and o == len(data["options"]) - 1
                ):
                    batch_features = {
                        key: torch.Tensor(
                            batch_features[key]).long().to(self.device)
                        for key in batch_features
                    }
                    with torch.no_grad():
                        output = self.model(**batch_features)
                    labels = (
                        np.array(batch_features["labels"].cpu()) != -100).astype(int)
                    label_position = np.argmax(labels, axis=1)
                    # option_score = output.logits[:,label_position,:]
                    for i in range(len(label_position)):
                        value_list.extend(
                            output.logits[i, label_position[i], option_token_ids[i]].tolist())
                option2value_list.append(option2value)

        # 关联选项及分数
        for option2value in option2value_list:
            for option in option2value.keys():
                option2value[option] = value_list[option2value[option]]

        reverse = True if option_rank == "logit" else False  # 按照loss/logit判断排序规则
        answers = [
            sorted(option2value.items(),
                   key=lambda x: x[1], reverse=reverse)[0][0]
            for option2value in option2value_list
        ]
        for data, ans in zip(datas, answers):
            data["predictions"] = [ans]

        # 计算选项的分数用于calibration等指标计算
        if likelihood:
            batch_features = {}
            mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
            for i, data in enumerate(datas):
                data_transformed = {}
                data_transformed["input"] = "Question: " + \
                    data["input"] + "\nChoices:"
                data_transformed["output"] = "A"
                for o, option in enumerate(data["options"]):
                    data["output"] = option
                    data_transformed["input"] += "\n" + "{}、{}".format(
                        alphabet[o], option
                    )
                features = GLMInstructionDataset.build_feature_from_sample(
                    data,
                    self.tokenizer,
                    max_length=max_length,
                    gpt_data=self.training_args.get("gpt_model", False),
                    mask_id=mask_id,
                    old_version_tokenizer=self.is_oldest_version,
                    eos_token=self.eos_token,
                    left_truncate=left_truncate,
                    rotary_type=self.rotary_type,
                )
                if batch_features:
                    for key, value in features.items():
                        batch_features[key].append(value)
                else:
                    batch_features = {key: [value]
                                      for key, value in features.items()}
            batch_features = {
                key: torch.Tensor(batch_features[key]).long().to(self.device)
                for key in batch_features
            }
            output = self.model(**batch_features)
            for i, data in enumerate(datas):
                label_index = (
                    (np.array(batch_features["labels"].cpu()) != -100)
                    .tolist()[i]
                    .index(True)
                )
                scores = []
                for i in range(len(data["options"])):
                    choice = alphabet[i]
                    choice_id = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(choice)
                    )[0]
                    score = torch.unsqueeze(
                        output.logits[0][label_index][choice_id], -1
                    )
                    scores.append(score)
                scores = torch.softmax(torch.cat(scores, -1), -1).tolist()
                data["likelihood"] = dict(zip(data["options"], scores))

        return datas

    def answer(
        self,
        query,
        num_beams=1,
        temperature=1,
        top_k=50,
        top_p=1,
        do_sample=False,
        left_truncate=False,
        max_output_length=-1,
        **gen_kwargs,
    ):
        """
        回答函数，让AntGLM根据给定的输入进行回答，并返回答案

        Args:
            query (string):
                AntGLM大模型的输入文本
            num_beams (int):
                解码过程所使用的集束搜索(beamsearch)的集束大小，通常情况下`num_beams`越大，
                生成结果越接近全局最优，但是会带来额外`num_beams`倍的计算开销。
                默认`num_beams=1`，即不使用集束搜索策略。
            temperature (int):
                采样生成策略中使用的温度参数，
                `temperature`越高采样的分布会越平滑，越低分布越尖锐。
            top_k (int):
                采样生成策略中使用的`top_k`采样参数，每一步解码将从top K个候选输出中选择生成。
            top_p (float):
                采样生成策略中使用的`top_p`采样参数，每一步解码将从累计概率为`P`的多个候选输出中选择生成。
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断
            max_output_length (int):
                最大输出长度

        Example::
        >>> path = "path-to-glm"
        >>> bot = GLMForInference(path)
        >>> text = "请问北京在哪里？"
        >>> answer = bot.answer(text)
        >>> print(answer)
        """
        data = {"input": query}
        max_length = self.training_args.get("max_length", 1024)
        max_output_length = (
            max_length - 2 if max_output_length == -1 else max_output_length
        )
        assert max_output_length < max_length
        if self.use_long_glm is True:
            dataset_class = GLMFoTDataset
            max_input_length = self.training_args.get("max_length", 1024)
        else:
            dataset_class = GLMInstructionDataset
            max_input_length = max_length - 4
        max_output_length, inputs = dataset_class.build_feature_from_sample(
            data,
            self.tokenizer,
            max_length=max_length,
            max_input_length=max_input_length,
            max_output_length=max_output_length,
            mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
            for_generation=True,
            left_truncate=left_truncate,
            eos_token=self.eos_token,
            gpt_data=self.training_args.get("gpt_model", False),
            old_version_tokenizer=self.is_oldest_version,
            rotary_type=self.rotary_type,
        )
        inputs = inputs.to(self.device)
        special_tokens = [
            self.tokenizer.eop_token,
            self.tokenizer.sop_token,
            self.tokenizer.eos_token,
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.eop_token_id,
            **gen_kwargs,
        )
        sop_index = outputs[0].tolist().index(self.tokenizer.sop_token_id)
        output_tokens = outputs[0][sop_index + 1:]
        output = self.tokenizer.decode(output_tokens)
        answer = self._post_process(output, special_tokens=special_tokens)
        return answer

    def generate(
            self,
            prompt,
            num_beams=1,
            temperature=1,
            top_k=50,
            top_p=1,
            do_sample=False,
            left_truncate=False,
            max_output_tokens=-1,
            num_return_sequences=1,
            **gen_kwargs,
    ) -> CompletionOutput:
        """
        回答函数，让AntGLM根据给定的输入进行回答，可能返回多条答案

        Args:
            prompt (string):
                AntGLM大模型的输入文本
            num_beams (int):
                解码过程所使用的集束搜索(beamsearch)的集束大小，通常情况下`num_beams`越大，
                生成结果越接近全局最优，但是会带来额外`num_beams`倍的计算开销。
                默认`num_beams=1`，即不使用集束搜索策略。
            temperature (int):
                采样生成策略中使用的温度参数，
                `temperature`越高采样的分布会越平滑，越低分布越尖锐。
            top_k (int):
                采样生成策略中使用的`top_k`采样参数，每一步解码将从top K个候选输出中选择生成。
            top_p (float):
                采样生成策略中使用的`top_p`采样参数，每一步解码将从累计概率为`P`的多个候选输出中选择生成。
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断
            max_output_tokens (int):
                每条生成内容，最大输出token数
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams

        Example::
        >>> path = "path-to-glm"
        >>> bot = GLMForInference(path)
        >>> text = "请问北京在哪里？"
        >>> answers = bot.generate(text)
        >>> print(answers.texts[0])
        """
        data = {"input": prompt}
        max_length = self.training_args.get("max_length", 1024)
        max_output_tokens = (
            max_length - 2 if max_output_tokens == -1 else max_output_tokens
        )
        assert max_output_tokens < max_length

        if self.use_long_glm is True:
            dataset_class = GLMFoTDataset
            max_input_length = self.training_args.get("max_length", 1024)
        else:
            dataset_class = GLMInstructionDataset
            max_input_length = max_length - 4

        max_output_tokens, inputs = dataset_class.build_feature_from_sample(
            data,
            self.tokenizer,
            max_length=max_length,
            max_input_length=max_input_length,
            max_output_length=max_output_tokens,
            mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
            for_generation=True,
            left_truncate=left_truncate,
            eos_token=self.eos_token,
            gpt_data=self.training_args.get("gpt_model", False),
            old_version_tokenizer=self.is_oldest_version,
            rotary_type=self.rotary_type,
        )
        inputs = inputs.to(self.device)
        special_tokens = [
            self.tokenizer.eop_token,
            self.tokenizer.sop_token,
            self.tokenizer.eos_token,
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.eop_token_id,
            num_return_sequences=num_return_sequences,
            **gen_kwargs,
        )
        texts = []
        reasons = []
        for output in outputs:
            sop_index = output.tolist().index(self.tokenizer.sop_token_id)
            output = output[sop_index + 1:]

            output_token = self.tokenizer.decode(output)
            output_str = self._post_process(
                output_token, special_tokens=special_tokens)
            texts.append(output_str)
            if output[-1].item() == self.tokenizer.eop_token_id:
                reasons.append(GenFinishReason.EOS)
            else:
                reasons.append(GenFinishReason.MAX_LEN)
        return CompletionOutput(texts=texts, finish_reasons=reasons)

    def generate_batch(
            self,
            prompts: List[str],
            num_beams: int = 1,
            temperature: int = 1,
            top_k: int = 50,
            top_p: float = 1.,
            do_sample: bool = False,
            max_output_tokens: int = -1,
            left_truncate: bool = False,
            num_return_sequences: int = 1,
            **gen_kwargs
    ) -> List[CompletionOutput]:
        """
        批量回答函数，让AntGLM根据给定的输入进行批量回答，并返回对应答案
        使用说明，参考文档 https://huggingface.co/blog/how-to-generate

        Args:
            prompts (List[string]):
                AntGLM大模型的输入文本列表。
            max_output_tokens (int):
                最大输出token数，平均1个token约 1.6 个字
            num_beams (int):
                解码过程所使用的集束搜索(beamsearch)的集束大小，通常情况下`num_beams`越大，
                生成结果越接近全局最优，但是会带来额外`num_beams`倍的计算开销。
                默认`num_beams=1`，即不使用集束搜索策略。
            temperature (int):
                采样生成策略中使用的温度参数，
                `temperature`越高采样的分布会越平滑，越低分布越尖锐。
            top_k (int):
                采样生成策略中使用的`top_k`采样参数，每一步解码将从top K个候选输出中选择生成。
            top_p (float):
                采样生成策略中使用的`top_p`采样参数，每一步解码将从累计概率为`P`的多个候选输出中选择生成。
            do_sample (boolean):
                是否使用采样生成策略，采样生成策略会给输出带来多样性，
                同时也会带来不确定行，请根据实际任务配置，默认不使用。
            left_truncate (boolean):
                是否进行左侧进行截断，默认超长输入则进行右侧截断
            num_return_sequences (int):
                返回的文本条数，需要确保 num_return_sequences <= num_beams
            gen_kwargs:
                更多参数见 https://huggingface.co/docs/transformers/main_classes/text_generation

        Example:
        >>> path = "path-to-glm"
        >>> bot = GLMForInference(path)
        >>> texts = [
        >>>     "请问北京在哪里？",
        >>>     "中国的首都在哪里？"
        >>> ]
        >>> answers = bot.generate_batch(texts)
        >>> print(answers)
        """
        if self.use_long_glm is True:
            raise NotImplementedError(
                "Long AntGLM is not support for batch generate.")

        max_input_tokens = self.training_args.get('max_length', 1024) - 2
        if max_output_tokens == -1:
            max_output_tokens = self.training_args.get('max_length', 1024) - 2
        if max_output_tokens > self.training_args.get('max_length', 1024) - 2:
            max_output_tokens = self.training_args.get('max_length', 1024) - 2
        inputs = self.get_batch_answer_inputs(
            prompts, left_truncate, max_input_tokens, max_output_tokens)
        inputs = inputs.to(self.device)
        special_tokens = [
            self.tokenizer.eop_token,
            self.tokenizer.sop_token,
            self.tokenizer.eos_token,
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_output_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.eop_token_id,
            num_return_sequences=num_return_sequences,
            **gen_kwargs,
        )
        # outputs: [batch * num_return_sequences, length]
        batch_out = []
        for i in range(len(prompts)):
            texts = []
            reasons = []
            for j in range(num_return_sequences):
                seq_id = outputs[i * num_return_sequences + j]
                sop_index = seq_id.tolist().index(self.tokenizer.sop_token_id)
                seq_id = seq_id[sop_index + 1:]
                seq_token = self.tokenizer.decode(seq_id)
                seq_str = self._post_process(
                    seq_token, special_tokens=special_tokens)
                texts.append(seq_str)
                if seq_id[-1].item() == self.tokenizer.eop_token_id:
                    reasons.append(GenFinishReason.EOS)
                else:
                    reasons.append(GenFinishReason.MAX_LEN)
            batch_out.append(CompletionOutput(
                texts=texts, finish_reasons=reasons))
        return batch_out

    def switch_answer(
        self,
        query,
        adapter_name="default",
        num_beams=1,
        temperature=1,
        top_k=50,
        top_p=1,
        do_sample=False,
        left_truncate=False,
        max_output_length=-1,
        **gen_kwargs,
    ):
        """
        多Peft模型的回答函数，让AntGLM根据给定的输入以及peft模块生成回答，并返回答案。
        该函数支持`peft==0.3.0`及以上版本，

        Args:
            adapter_name (string):
                需要使用的peft模块的名字，对应于模型加载目录中`adapters`目录下每个字目录的名字。

        Example::
        给定输入目录，可以通过以下方式进行输出
        ```
            |-- path-to-glm-with-multi-lora
            |   |-- adapters
            |   |   |-- test_adapter1
            |   |   |   |-- xxx
            |   |   |-- test_adapter2
            | xxx
        ```
        >>> path = "path-to-glm-with-multi-lora"
        >>> bot = GLMForInference(path)
        >>> text = "请问北京在哪里？"
        >>> answer = bot.switch_answer(text, adapter_name="test_adapter1")
        >>> print(answer)
        """
        self.model.set_adapter(adapter_name)
        answer = self.answer(
            query, num_beams=num_beams, temperature=temperature,
            top_k=top_k, top_p=top_p, do_sample=do_sample,
            left_truncate=left_truncate, max_output_length=max_output_length,
            **gen_kwargs,
        )
        return answer

    def get_batch_answer_inputs(
            self,
            querys: List[str],
            left_truncate: bool = False,
            max_input_length: int = 1022,
            max_output_length: int = 1022) -> BatchEncoding:
        batch_size = len(querys)
        input_ids, position_ids, generation_attention_mask = [], [], []

        max_length = self.training_args.get('max_length', 1024)
        max_output_length = max_length - 2 if max_output_length == -1 else max_output_length
        assert max_output_length < max_length
        for query in querys:
            data = {"input": query}
            max_output_length, instance_input = GLMInstructionDataset.build_feature_from_sample(
                data, self.tokenizer,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
                for_generation=True,
                left_truncate=left_truncate,
                gpt_data=self.training_args.get("gpt_model", False),
                eos_token=self.eos_token,
                old_version_tokenizer=self.is_oldest_version,
                rotary_type=self.rotary_type,
            )
            input_ids.append(instance_input["input_ids"])
            position_ids.append(instance_input["position_ids"])
            generation_attention_mask.append(
                instance_input["generation_attention_mask"]
            )

        max_ids_length = max([input.size(1) for input in input_ids])

        if self.training_args.get("gpt_model", False):
            tmp_max_output_length = max_length - max_ids_length - 2
            max_output_length = min(max_output_length, tmp_max_output_length)

        for i in range(batch_size):
            cur_ids_length = input_ids[i].size(1)
            if cur_ids_length < max_ids_length:
                # pad input ids
                pad_input_ids = input_ids[i].new_zeros(
                    (1, max_ids_length - cur_ids_length)
                )
                input_ids[i] = torch.cat([pad_input_ids, input_ids[i]], dim=-1)

                # pad postition ids with left pad
                # 0, 1, 2, 3, 4 ... -> 0, ..., 0, 1, 2, 3, 4, ...
                pad_position_ids = input_ids[i].new_zeros(
                    (1, 2, max_ids_length - cur_ids_length)
                )
                position_ids[i] = torch.cat(
                    [pad_position_ids, position_ids[i]], dim=-1)

                # pad generation attention mask with left and bottom pad
                new_attention_mask = input_ids[i].new_zeros(
                    1,
                    1,
                    max_ids_length + max_output_length,
                    max_ids_length + max_output_length,
                )
                new_attention_mask[
                    :,
                    :,
                    max_ids_length - cur_ids_length:,
                    max_ids_length - cur_ids_length:,
                ] = generation_attention_mask[i]
                generation_attention_mask[i] = new_attention_mask.contiguous()

        input_ids = torch.cat(input_ids, dim=0)
        position_ids = torch.cat(position_ids, dim=0)
        generation_attention_mask = torch.cat(generation_attention_mask, dim=0)

        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "generation_attention_mask": generation_attention_mask,
        }

        inputs = BatchEncoding(inputs)
        return inputs

    @torch.no_grad()
    def batch_answer_with_loss(
        self,
        datas: List[dict],
        max_input_length: int = -1,
        max_output_length: int = -1,
        left_truncate: bool = False
    ) -> List[str]:
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
        max_length = self.training_args.get("max_length", 1024)
        if max_input_length != -1 and max_output_length != -1:
            data_max_length = max_input_length + max_output_length + 4
            if self.use_long_glm is True:
                max_length = data_max_length
            else:
                max_length = data_max_length if data_max_length < max_length else max_length
        if max_output_length == -1:
            max_output_length = 2 * max_length

        if self.use_long_glm is True:
            dataset_class = GLMFoTDataset
            max_input_length = self.training_args.get("max_length", 1024)
        else:
            dataset_class = GLMInstructionDataset
            max_input_length = max_length - 4

        for i, data in enumerate(datas):
            mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
            features = dataset_class.build_feature_from_sample(
                data,
                self.tokenizer,
                max_length=max_length,
                max_input_length=max_input_length,
                max_output_length=max_output_length,
                gpt_data=self.training_args.get("gpt_model", False),
                mask_id=mask_id,
                eos_token=self.eos_token,
                old_version_tokenizer=self.is_oldest_version,
                left_truncate=left_truncate,
                rotary_type=self.rotary_type,
            )

            features["attention_mask"] = [features["attention_mask"]]
            batch_features = {key: [value] for key, value in features.items()}
            batch_features = {
                key: torch.Tensor(
                    batch_features[key]).long().to(self.device)
                for key in batch_features
            }

            output = self.model(**batch_features)
            loss = loss_fct(
                output["logits"].view(-1, output["logits"].size(-1)),
                batch_features["labels"].view(-1),
            )
            loss = loss[torch.nonzero(loss)]
            data["loss"] = loss.view(-1).cpu().tolist()

            # Support the format check for evaluation
            data["predictions"] = [""]
            data["references"] = [""]
        return datas

    def batch_answer(
            self,
            datas: List[dict],
            num_beams: int = 1,
            temperature: int = 1,
            top_k: int = 50,
            top_p: float = 1.,
            do_sample: bool = False,
            max_input_length: int = -1,
            max_output_length: int = -1,
            left_truncate: bool = False) -> List[str]:
        """
        批量回答函数，让AntGLM根据给定的输入进行批量回答，并返回对应答案

        Args:
            query (string, List[string]):
                AntGLM大模型的输入文本，可以是一条单独的文本，也可以一个文本列表。

        Example::
        >>> path = "path-to-glm"
        >>> bot = GLMForInference(path)
        >>> texts = [
        >>>     "请问北京在哪里？",
        >>>     "中国的首都在哪里？"
        >>> ]
        >>> answers = bot.answer(texts)
        >>> print(answers)
        """
        if self.use_long_glm is True:
            raise NotImplementedError(
                "Long AntGLM is not support for batch answer.")

        if max_input_length == -1:
            max_input_length = self.training_args.get('max_length', 1024) - 2
        if max_input_length > self.training_args.get('max_length', 1024) - 2:
            max_input_length = self.training_args.get('max_length', 1024) - 2
        if max_output_length == -1:
            max_output_length = self.training_args.get('max_length', 1024) - 2
        if max_output_length > self.training_args.get('max_length', 1024) - 2:
            max_output_length = self.training_args.get('max_length', 1024) - 2
        querys = [data["input"] for data in datas]
        inputs = self.get_batch_answer_inputs(
            querys, left_truncate, max_input_length, max_output_length)
        inputs = inputs.to(self.device)
        special_tokens = [
            self.tokenizer.eop_token,
            self.tokenizer.sop_token,
            self.tokenizer.eos_token,
        ]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.eop_token_id,
        )
        outputs = outputs.tolist()
        for i in range(len(outputs)):
            sop_index = outputs[i].index(self.tokenizer.sop_token_id)
            outputs[i] = outputs[i][sop_index + 1:]
        output = [self.tokenizer.decode(o) for o in outputs]
        answer = [self._post_process(
            o, special_tokens=special_tokens) for o in output]
        for data, ans in zip(datas, answer):
            data["predictions"] = [ans]
        return datas

    def generate_stream(
        self,
        prompt,
        num_beams=1,
        temperature=1,
        top_k=50,
        top_p=1,
        left_truncate=False,
        max_output_tokens=-1,
        do_sample=False,
        **gen_kwargs,
    ):
        """
        流式回答函数，让AntGLM根据给定的输入进行流式回答，逐步返回生成的文本。

        """
        data = {"input": prompt}
        max_length = self.training_args.get("max_length", 1024)
        max_output_tokens = (
            max_length - 2 if max_output_tokens == -1 else max_output_tokens
        )
        assert max_output_tokens < max_length
        max_output_tokens, inputs = GLMInstructionDataset.build_feature_from_sample(
            data,
            self.tokenizer,
            max_length=max_length,
            max_input_length=max_length - 2,
            max_output_length=max_output_tokens,
            mask_id=self.tokenizer.convert_tokens_to_ids(self.mask),
            for_generation=True,
            left_truncate=left_truncate,
            eos_token=self.eos_token,
            old_version_tokenizer=self.is_oldest_version,
            rotary_type=self.rotary_type,
        )
        inputs = inputs.to(self.device)
        # special_tokens = [self.tokenizer.eop_token,
        #                   self.tokenizer.sop_token, self.tokenizer.eos_token]
        id_list = []
        pre_str = ""
        for output in self.model.generate_stream(
            **inputs,
            max_new_tokens=max_output_tokens,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.tokenizer.eop_token_id,
            do_sample=do_sample,
            **gen_kwargs,
        ):
            # output = output.tolist()[0][len(inputs["input_ids"][0]):]
            output = output.tolist()[0]
            id_list += output
            total_response = self.tokenizer.decode(id_list)
            # todo: need a better way for oov
            if total_response and total_response[-1] != '�':
                response = total_response[len(pre_str):]
                pre_str = total_response
                yield response

    def reset_session(self):
        self.chat_history = ""
        self.turn = 1

    def chat(self, query, num_beams, temperature, top_k, top_p, do_sample=False):
        new_query = self.chat_history + f"第{self.turn}轮\n用户: {query}\n机器人:"
        output = self.answer(
            new_query,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            left_truncate=True,
            do_sample=do_sample,
        )
        self.chat_history = new_query + output + '\n'
        self.turn += 1
        return output

    @contextmanager
    def disable_adapter(self):
        from solutions.antllm.antllm.models.peft.modeling_peft import PeftModel
        if isinstance(self.model, PeftModel):
            with self.model.disable_adapter():
                yield self
        else:
            yield self

    def set_adapter(self, adapter_name):
        self.model.set_adapter(adapter_name)


class PeftGLMForInference(GLMForInference):
    def __init__(self, adapter_config: dict, *args, **kwargs):
        """
        GLMForInference默认要求所有的adapter统一放到一个父亲目录下，而maya上一定是分散在不同目录。
        即便如此，统一到一个目录下的要求有些定制化。因此封装这个接口支持adapter可存放在不同目录下。
        Args:
            adapter_config: dict, adapter名字/路径配置，例如：
            {
                "adapter_name1": ["path-to-adapter1"],
                "adapter_name2": ["path-to-adapter2"],
            }
        """
        super(PeftGLMForInference, self).__init__(*args, **kwargs)
        self.load_adapters(adapter_config)

    def load_adapters(self, adapter_config: dict):
        from solutions.antllm.antllm.models.peft.modeling_peft import (
            PeftModel,
        )  # noqa
        peft_init_flag = isinstance(self.model, PeftModel)
        for adapter_name, adapter_paths in adapter_config.items():
            print(f"Load maya adapter: {adapter_name}, {adapter_paths}")
            # lora weight on aistudio is a list, we only upload one for each adapter
            adapter_path = adapter_paths[0]
            if peft_init_flag is True:
                self.model.load_adapter(
                    adapter_path, adapter_name=adapter_name
                )
            else:
                peft_init_flag = True
                self.model = PeftModel.from_pretrained(
                    self.model, adapter_path, adapter_name=adapter_name)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_path",
        default="/disk5/xinyu.kxy/chatgpt/models/glm_2b_smask",
        type=str,
        action="store",
        help="model path"
    )
    parser.add_argument(
        "--lora_config_path",
        default=None,
        type=str,
        action="store",
        help="lora config path",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    bot = GLMForInference(args.model_path)
    __import__('pudb').set_trace()
    print(bot.answer("你好"))
    queries = [{'input': '北京在哪里'}, {'input': '中国在哪里'}]
    answers = bot.batch_answer(queries)
    print(answers)
    print(
        bot.batch_answer_with_options(
            [
                {"input": "你在哪里", "options": ["我不知道", "在北京"]},
                {"input": "月亮在哪里", "options": ["地球", "地球旁边"]},
            ],
            likelihood=True,
        )
    )
    for answers in bot.generate_stream("北京在哪里", 1, 0.2, 10, 0.95):
        print(answers)


if __name__ == "__main__":
    main()
