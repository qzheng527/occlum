import json
import logging
import os
import shutil

import torch
from dataclasses import dataclass, field
from peft import (  # noqa
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptLearningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model
)
from solutions.antllm.antllm.data.dataset.chatglm2_instruction_dataset import (
    ChatGLM2InstructionDataset
)
from solutions.antllm.antllm.data.dataset.chatglm_instruction_dataset import (
    ChatGLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.glm_embedding_dataset import (
    GLMEmbeddingDataset
)
from solutions.antllm.antllm.data.dataset.glm_fot_dataset import GLMFoTDataset
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.glm_packed_dataset import (
    GLMPackedDataset
)
from solutions.antllm.antllm.data.dataset.instruction_dataset import (
    InstructionDataset
)
from solutions.antllm.antllm.datav2.datasets import AutoDataset
from solutions.antllm.antllm.datav2.glm import GLMSeq2SeqDataset
from solutions.antllm.antllm.models.chatglm2.configuration_chatglm2 import (
    ChatGLM2Config
)
from solutions.antllm.antllm.models.chatglm2.modeling_chatglm2 import (
    ChatGLM2ForConditionalGeneration
)
from solutions.antllm.antllm.models.chatglm2.tokenization_chatglm2 import \
    ChatGLMTokenizer as ChatGLM2Tokenizer
from solutions.antllm.antllm.models.chatglm.configuration_chatglm import (
    ChatGLMConfig
)
from solutions.antllm.antllm.models.chatglm.modeling_chatglm import (
    ChatGLMForConditionalGeneration
)
from solutions.antllm.antllm.models.chatglm.tokenization_chatglm import (
    ChatGLMTokenizer
)
from solutions.antllm.antllm.models.embeddings.modeling_embedding import (
    GLMForEmbedding
)
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.models.glm.modeling_glm import (
    GLMForConditionalGeneration
)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.peft.modeling_peft import (
    AntPeftForCausalLM
)
from solutions.antllm.antllm.models.peft.tuner import (
    AdaLoraConfig,
    PeftBitfitConfig,
    PeftROEMConfig,
    UniPELTConfig
)
from solutions.antllm.antllm.models.peft.utils import (
    prepare_model_for_kbit_training
)
from solutions.antllm.antllm.training.trainer.sft_trainer import SFTTrainer
from solutions.antllm.antllm.utils import mpu
from solutions.antllm.antllm.utils.utils import is_yaml
from solutions.antllm.antllm.utils.version_utils import is_oldest_version
from transformers import (  # noqa
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerCallback


class DynamicPaddingCollator(DataCollatorMixin):
    def __init__(self, pad_id=None):
        super().__init__()
        self.pad_id = pad_id

    def __call__(self, features):
        if self.pad_id is not None:
            max_length = -1
            for row in range(len(features)):
                input_ids = features[row]['input_ids']
                for col in range(input_ids.shape[0]):
                    if input_ids[col].item() == self.pad_id:
                        break
                max_length = col if col > max_length else max_length
            for row in range(len(features)):
                features[row]['input_ids'] = features[row]['input_ids'][:max_length]
                features[row]['labels'] = features[row]['labels'][:max_length]
                # support for ChatGLM2
                if len(features[row]['position_ids'].shape) == 2:
                    features[row]['position_ids'] = features[row]['position_ids'][:, :max_length]
                else:
                    features[row]['position_ids'] = features[row]['position_ids'][:max_length]

                if len(features[row]['attention_mask'].shape) == 3:
                    features[row]['attention_mask'] = features[row]['attention_mask'][
                        :, :max_length, :max_length]
                # support for ChatGLM2
                elif len(features[row]['attention_mask'].shape) == 1:
                    features[row]['attention_mask'] = features[row]['attention_mask'][:max_length]
        else:
            print('Ignore dynamic_padding, while dynamic_padding, pad_id muast be set')
        batch = {}
        for feature in features:
            for key in feature:
                if key not in batch:
                    batch[key] = []
                batch[key].append(feature[key].unsqueeze(0))
        batch = {key: torch.cat(value) for key, value in batch.items()}
        return batch


class SaveCallback(TrainerCallback):
    def __init__(self, files_to_save, args_to_save={}):
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
        os.makedirs(artifact_path, exist_ok=True)
        json.dump(self.args_to_save, open(os.path.join(artifact_path,
                  'hyper_parameters.json'), 'w'), ensure_ascii=False, indent=2)
        for name in self.files_to_save:
            if not os.path.exists(name):
                continue
            try:
                if os.path.isfile(name):
                    shutil.copy(name, artifact_path)
                elif os.path.isdir(name):
                    shutil.copytree(name, os.path.join(
                        artifact_path, os.path.basename(name)))
            except Exception:
                continue


class EpochSaveCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        orig_output_dir = args.output_dir
        orig_no_save_deepspeed_checkpoint = self.trainer.no_save_deepspeed_checkpoint
        orig_save_total_limit = args.save_total_limit
        self.trainer.args.output_dir = os.path.join(
            args.output_dir, 'epochs')
        self.trainer.no_save_deepspeed_checkpoint = True
        self.trainer.args.save_total_limit = None
        self.trainer._save_checkpoint(self.trainer.model, None, metrics=None)
        self.trainer.callback_handler.on_save(args, state, control)
        self.trainer.args.output_dir = orig_output_dir
        self.trainer.no_save_deepspeed_checkpoint = orig_no_save_deepspeed_checkpoint
        self.trainer.args.save_total_limit = orig_save_total_limit


@dataclass
class ModelArguments:
    pretrained_model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lm_type: str = field(
        metadata={"help": "Type of language model"}
    )
    no_save_deepspeed_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to save deepspeed checkpoint."})
    no_save_base_model: bool = field(
        default=False, metadata={"help": "Whether to save base model."})


@dataclass
class PeftArguments:
    peft_type: str = field(
        default=None, metadata={"help": "Whether use peft"})
    prompt_init_text: str = field(
        default=None, metadata={"help": "The init text of prompt learning"})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The rank for lora, qlora and unipelt training."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={
            "help": "The lora parameters scale factor for lora, qlora and unipelt training."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for lora, qlora and unipelt training."}
    )
    adalora_init_rank: int = field(
        default=40,
        metadata={"help": "The init rank for adalora model training."}
    )
    adalora_target_rank: int = field(
        default=32,
        metadata={"help": "The final target rank for adalora model training."}
    )
    num_virtual_tokens: int = field(
        default=20,
        metadata={
            "help": "The number of virtual tokens for puting, prompt and prefix-tuning."}
    )
    encoder_hidden_size: int = field(
        default=128,
        metadata={"help": "The encoder hidden size for prompt and prefix-tuning."}
    )
    ptuning_encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The encoder layer num for ptuning training."}
    )
    ptuning_encoder_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout ratio for ptuning training."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )


@dataclass
class DataArguments:
    max_length: int = field(
        metadata={"help": "The maximum total input sequence length"}
    )
    max_input_length: int = field(
        metadata={"help": "The maximum total prompt length"}
    )
    max_output_length: int = field(
        metadata={"help": "The maximum total output length"}
    )
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Path to test data"}
    )
    eos_token: str = field(
        default='<|endoftext|>', metadata={"help": "end token to use"}
    )
    no_append_glm_mask: bool = field(
        default=False, metadata={"help": "Whether to append glm_mask"})
    glm_mask: str = field(
        default='[gMASK]', metadata={"help": "Mask to use in glm"}
    )
    gpt_data: bool = field(
        default=False, metadata={"help": "Whether train gpt_data"})

    left_truncate: bool = field(
        default=False, metadata={"help": "Whether truncate at the left side"})

    dynamic_padding: bool = field(
        default=False, metadata={"help": "Whether dynamically padding in each batch"})

    undirectional_attention: bool = field(
        default=False, metadata={"help": "undirectional attention"})

    use_packed_data: bool = field(
        default=False, metadata={"help": "Use packed dataset for training."}
    )

    use_long_glm: bool = field(
        default=False, metadata={"help": "Use Fot dataset for training."}
    )

    long_glm_factor: int = field(
        default=2, metadata={"help": "The scale factor to expand the max data length."}
    )

    isolation_position_ids: bool = field(
        default=False, metadata={"help": "Whether isolate position ids in pack training manner"})

    add_cls: bool = field(
        default=True, metadata={"help": "Whether add [CLS] token in each pack training manner"})

    online_packed: bool = field(
        default=True, metadata={"help": "Whether use online pack data process"})

    datasetv2: bool = field(
        default=False, metadata={"help": "Whether to use dataset v2"})

    rotary_type: str = field(
        default="none", metadata={"help": "Which rotary method used in data construct"})


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PeftArguments))
    model_args, data_args, training_args, peft_args = parser.parse_args_into_dataclasses()
    logger = logging.getLogger(__name__)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))

    mpu.initialize_model_parallel(1)
    # podname = os.environ.get('ILOGTAIL_PODNAME', 'master')

    # if 'master' in podname:
    #     save_checkpoint = True
    # elif 'worker' in podname:
    #     save_checkpoint = False
    # else:
    #     save_checkpoint = True
    if training_args.resume_from_checkpoint == 'true':
        logger.info(f'Resume from {training_args.output_dir}')
        resume_from_checkpoint = True
    else:
        logger.info('Train from scratch')
        resume_from_checkpoint = False
    logger.info(f'world_size: {world_size}, global_rank: {global_rank}')

    train_data_path = data_args.train_data
    test_data_path = data_args.test_data
    lm_type = model_args.lm_type

    old_version_tokenizer = False
    if lm_type == 'seq2seq':
        auto_model_class = GLMForConditionalGeneration  # noqa
        auto_config_class = GLMConfig
        if data_args.use_long_glm is True:
            dataset_class = GLMFoTDataset
            data_args.max_length *= data_args.long_glm_factor
        elif data_args.use_packed_data is True:
            dataset_class = GLMPackedDataset
        else:
            dataset_class = GLMInstructionDataset
        if is_oldest_version(model_args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import \
                GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            data_args.glm_mask = '[sMASK]'
            old_version_tokenizer = True
        else:
            auto_tokenization_class = GLMTokenizer
            old_version_tokenizer = False
    elif lm_type == "chatglm":
        auto_model_class = ChatGLMForConditionalGeneration
        auto_config_class = ChatGLMConfig
        dataset_class = ChatGLMInstructionDataset
        auto_tokenization_class = ChatGLMTokenizer
    elif lm_type == "chatglm2":
        auto_model_class = ChatGLM2ForConditionalGeneration
        auto_config_class = ChatGLM2Config
        dataset_class = ChatGLM2InstructionDataset
        auto_tokenization_class = ChatGLM2Tokenizer
    elif lm_type == "embedding":
        auto_model_class = GLMForEmbedding
        auto_config_class = GLMConfig
        dataset_class = GLMEmbeddingDataset
        if is_oldest_version(model_args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import \
                GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            data_args.glm_mask = '[sMASK]'
            old_version_tokenizer = True
        else:
            auto_tokenization_class = GLMTokenizer
            old_version_tokenizer = False
    else:
        auto_model_class = AutoModelForCausalLM  # noqa
        auto_config_class = AutoConfig
        dataset_class = InstructionDataset
        auto_tokenization_class = AutoTokenizer

    # logger.info('Build model', ranks=[0])
    local_rank = global_rank % torch.cuda.device_count()
    device = {'': local_rank if torch.cuda.is_available() else 'cpu'}
    logger.info(
        f'world_size: {world_size}, global_rank: {global_rank}, local_rank: {local_rank}, device: {device}')
    tokenizer = auto_tokenization_class.from_pretrained(  # noqa
        model_args.pretrained_model_name_or_path, trust_remote_code=True,
    )

    if data_args.use_long_glm is True:
        config = auto_config_class.from_pretrained(
            model_args.pretrained_model_name_or_path)
        config.focused_attention = True
        config.use_cache = True
    else:
        config = None

    if peft_args.peft_type == "qlora":
        import bitsandbytes as bnb  # noqa
        from transformers import BitsAndBytesConfig

        data_type = torch.float16 if training_args.fp16 else torch.bfloat16 if training_args.bf16 else torch.float32

        model = auto_model_class.from_pretrained(  # noqa
            model_args.pretrained_model_name_or_path, trust_remote_code=True, device_map=device,
            config=config,
            load_in_4bit=peft_args.bits == 4,
            load_in_8bit=peft_args.bits == 8,
            torch_dtype=data_type,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=peft_args.bits == 4,
                load_in_8bit=peft_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=data_type,
                bnb_4bit_use_double_quant=peft_args.double_quant,
                bnb_4bit_quant_type=peft_args.quant_type,
            )
        )
    else:
        model = auto_model_class.from_pretrained(  # noqa
            model_args.pretrained_model_name_or_path, config=config, trust_remote_code=True, device_map=device)

    # Load peft model
    if peft_args.peft_type:
        peft_task_type = TaskType.ANT_EMBEDDING if lm_type == "embedding" else TaskType.ANT_CAUSAL_LM

        if peft_args.peft_type in ["lora", "qlora"]:
            if peft_args.peft_type == "qlora":
                model = prepare_model_for_kbit_training(
                    model, training_args.gradient_checkpointing)

            peft_config = LoraConfig(
                task_type=peft_task_type,
                inference_mode=False,
                r=peft_args.lora_rank,
                lora_alpha=peft_args.lora_alpha,
                lora_dropout=peft_args.lora_dropout
            )
        elif peft_args.peft_type == "unipelt":
            peft_config = UniPELTConfig(
                task_type=peft_task_type,
                inference_mode=False,
                r=peft_args.lora_rank,
                lora_alpha=peft_args.lora_alpha,
                lora_dropout=peft_args.lora_dropout
            )
        elif peft_args.peft_type == "ptuning":
            peft_config = PromptEncoderConfig(
                task_type=peft_task_type,
                inference_mode=False,
                encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                encoder_num_layers=peft_args.ptuning_encoder_num_layers,
                encoder_dropout=peft_args.ptuning_encoder_dropout,
                encoder_hidden_size=peft_args.encoder_hidden_size,
                num_virtual_tokens=peft_args.num_virtual_tokens
            )
        elif peft_args.peft_type == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=peft_task_type,
                inference_mode=False,
                num_virtual_tokens=peft_args.num_virtual_tokens,
                prefix_projection=True,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=model.config.num_layers,
                encoder_hidden_size=peft_args.encoder_hidden_size,
                token_dim=model.config.hidden_size
            )
        elif peft_args.peft_type == "prompt":
            peft_config = PromptTuningConfig(
                task_type=peft_task_type,
                inference_mode=False,
                prompt_tuning_init=PromptTuningInit.TEXT if peft_args.prompt_init_text else PromptTuningInit.RANDOM,
                num_virtual_tokens=peft_args.num_virtual_tokens,
                prompt_tuning_init_text=peft_args.prompt_init_text if peft_args.prompt_init_text else None,
                tokenizer_name_or_path=model_args.pretrained_model_name_or_path
            )
        elif peft_args.peft_type == "roem":
            peft_config = PeftROEMConfig(
                task_type=peft_task_type,
                inference_mode=False
            )
        elif peft_args.peft_type == "bitfit":
            peft_config = PeftBitfitConfig(
                task_type=peft_task_type,
                inference_mode=False
            )
        elif peft_args.peft_type == "adalora":
            peft_config = AdaLoraConfig(
                task_type=peft_task_type,
                inference_mode=False,
                lora_alpha=peft_args.lora_alpha,
                init_r=peft_args.adalora_target_rank,
                target_r=peft_args.adalora_init_rank,
                lora_dropout=peft_args.lora_dropout,
            )
        else:
            raise ValueError(
                "The param 'peft_type' must in " +
                "['lora', 'ptuning', 'prefix', 'prompt', 'adalora', 'unipelt', 'roem', 'bitfit'], " +
                f"but get: {peft_args.peft_type}")
        logger.info(
            f"Load Peft {peft_args.peft_type} model ......")

        if isinstance(peft_config, PromptLearningConfig):
            logger.info(
                "User the prompt learning method, reduce the max length with virtual tokens.")
            data_args.max_length -= peft_args.num_virtual_tokens
            data_args.max_input_length -= peft_args.num_virtual_tokens // 2
            data_args.max_output_length -= peft_args.num_virtual_tokens // 2

        if resume_from_checkpoint is True:
            try:
                model = AntPeftForCausalLM.from_pretrained(
                    model, training_args.output_dir, is_trainable=True,
                    resume_from_checkpoint=resume_from_checkpoint
                )
                resume_from_checkpoint = False
                logger.info(
                    f"Resume the last checkpoint from {model.resume_ckpt_dir}")
            except Exception:
                logger.warning(
                    f"Can not find any ckpt in {training_args.output_dir}, "
                    "init a new peft model from config."
                )
                model = get_peft_model(model, peft_config)
                resume_from_checkpoint = False
        else:
            model = get_peft_model(model, peft_config)

        if peft_args.peft_type == "qlora":
            for name, module in model.named_modules():
                if 'norm' in name:
                    module = module.to(data_type)
                if 'word_embeddings' in name:
                    if hasattr(module, 'weight'):
                        if module.weight.dtype == torch.float32:
                            module = module.to(data_type)
        logger.info(
            f"Reduce trainalbe params:\n{model.print_trainable_parameters()}")

    files_to_save = ['merge.model', 'tokenizer_config.json']
    files_to_save = [os.path.join(
        model_args.pretrained_model_name_or_path, filename) for filename in files_to_save]

    if os.path.exists(os.path.join(model_args.pretrained_model_name_or_path, 'hyper_parameters.json')):
        pretrain_model_args = json.load(
            open(os.path.join(model_args.pretrained_model_name_or_path, 'hyper_parameters.json')))
        if 'eos_token' in pretrain_model_args:
            data_args.eos_token = pretrain_model_args['eos_token']
        if 'rotary_type' in pretrain_model_args:
            data_args.rotary_type = pretrain_model_args['rotary_type']

    # logger.info(f'Build data loader from path {train_data_path}', ranks=[0])
    test_dataset = None
    if data_args.datasetv2:
        if is_yaml(train_data_path):
            train_dataset = AutoDataset.from_config(train_data_path, tokenizer)
        else:
            train_dataset = GLMSeq2SeqDataset(name="train",
                                              data_path=train_data_path,
                                              tokenizer=tokenizer,
                                              max_length=data_args.max_length,
                                              max_input_length=data_args.max_input_length,
                                              max_output_length=data_args.max_output_length,
                                              left_truncate=data_args.left_truncate,
                                              scatter_num=world_size)
        if test_data_path is not None:
            if is_yaml(test_data_path):
                test_dataset = AutoDataset.from_config(test_data_path, tokenizer)
            else:
                test_dataset = GLMSeq2SeqDataset(name="test",
                                                 data_path=test_data_path,
                                                 tokenizer=tokenizer,
                                                 max_length=data_args.max_length,
                                                 max_input_length=data_args.max_input_length,
                                                 max_output_length=data_args.max_output_length,
                                                 left_truncate=data_args.left_truncate,
                                                 scatter_num=1)
    else:
        train_dataset = dataset_class(data_path=train_data_path,
                                      tokenizer=tokenizer,
                                      max_length=data_args.max_length,
                                      max_input_length=data_args.max_input_length,
                                      max_output_length=data_args.max_output_length,
                                      return_dict=False,
                                      no_append_glm_mask=data_args.no_append_glm_mask,
                                      gpt_data=data_args.gpt_data,
                                      left_truncate=data_args.left_truncate,
                                      world_size=world_size,
                                      global_rank=global_rank,
                                      shard_data=True,
                                      glm_mask=data_args.glm_mask,
                                      old_version_tokenizer=old_version_tokenizer,
                                      undirectional_attention=data_args.undirectional_attention,
                                      isolation_position_ids=data_args.isolation_position_ids,
                                      add_cls=data_args.add_cls,
                                      online_packed=data_args.online_packed,
                                      eos_token=data_args.eos_token,
                                      rotary_type=data_args.rotary_type
                                      )
        if test_data_path is not None:
            test_dataset = dataset_class(data_path=test_data_path,
                                         tokenizer=tokenizer,
                                         max_length=data_args.max_length,
                                         max_input_length=data_args.max_input_length,
                                         max_output_length=data_args.max_output_length,
                                         left_truncate=data_args.left_truncate,
                                         return_dict=False,
                                         shard_data=False,
                                         no_append_glm_mask=data_args.no_append_glm_mask,
                                         gpt_data=data_args.gpt_data,
                                         glm_mask=data_args.glm_mask,
                                         old_version_tokenizer=old_version_tokenizer,
                                         undirectional_attention=data_args.undirectional_attention,
                                         isolation_position_ids=data_args.isolation_position_ids,
                                         add_cls=data_args.add_cls,
                                         online_packed=data_args.online_packed,
                                         rotary_type=data_args.rotary_type,
                                         eos_token=data_args.eos_token,
                                         )

    data_collator = None
    if data_args.dynamic_padding:
        data_collator = DynamicPaddingCollator(
            pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    trainer = SFTTrainer(model=model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         callbacks=[SaveCallback(files_to_save, args_to_save={
                                                 'max_length': data_args.max_length if not data_args.use_long_glm else
                                                 data_args.max_length // data_args.long_glm_factor,
                                                 'peft_type': peft_args.peft_type,
                                                 'eos_token': data_args.eos_token,
                                                 'rotary_type': data_args.rotary_type,
                                                 'gpt_model': data_args.gpt_data})],
                         no_save_deepspeed_checkpoint=model_args.no_save_deepspeed_checkpoint,
                         save_pytorch_model_bin_checkpoint=True, rank=global_rank,
                         train_peft=True if peft_args.peft_type else False,
                         no_save_base_model=model_args.no_save_base_model
                         )
    trainer.add_callback(EpochSaveCallback(trainer))

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if not model_args.no_save_base_model:
            trainer.save_model()
        if peft_args.peft_type:
            model.save_pretrained(training_args.output_dir)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval and test_data_path is not None:
        trainer.evaluate()


if __name__ == '__main__':
    main()
