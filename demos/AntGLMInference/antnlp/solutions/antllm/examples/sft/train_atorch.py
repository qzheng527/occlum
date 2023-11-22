import json
import logging
import os
import shutil
import time
from datetime import timedelta

from packaging import version

import atorch
import peft
import torch
from atorch.utils.meta_model_utils import init_empty_weights_with_disk_offload
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
from peft.tuners.lora import LoraLayer
from solutions.antllm.antllm.data.dataset.chatglm_instruction_dataset import (
    ChatGLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.chatglm2_instruction_dataset import \
    ChatGLM2InstructionDataset
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
from solutions.antllm.antllm.models.chatglm.configuration_chatglm import (
    ChatGLMConfig
)
from solutions.antllm.antllm.models.chatglm.modeling_chatglm import (
    ChatGLMForConditionalGeneration
)
from solutions.antllm.antllm.models.chatglm.tokenization_chatglm import \
    ChatGLMTokenizer
from solutions.antllm.antllm.models.chatglm2.modeling_chatglm2 import \
    ChatGLM2ForConditionalGeneration
from solutions.antllm.antllm.models.chatglm2.tokenization_chatglm2 import \
    ChatGLMTokenizer as ChatGLM2Tokenizer
from solutions.antllm.antllm.models.chatglm2.configuration_chatglm2 import \
    ChatGLM2Config
from solutions.antllm.antllm.models.embeddings.modeling_embedding import \
    GLMForEmbedding
from solutions.antllm.antllm.models.glm.configuration_glm import GLMConfig
from solutions.antllm.antllm.models.glm.modeling_glm import GLMModel
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.peft.tuner import (
    AdaLoraConfig,
    PeftBitfitConfig,
    PeftROEMConfig,
    UniPELTConfig
)
from solutions.antllm.antllm.models.peft.utils import (
    prepare_model_for_kbit_training
)
from solutions.antllm.antllm.training.trainer.atorch_trainer import (
    STREAMING_CKPT_DIR,
    AtorchTrainer,
    get_last_checkpoint
)
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
from transformers.trainer_callback import TrainerCallback

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
try:
    import bitsandbytes as bnb  # noqa
except ImportError:
    bnb = None

# 在不同卡数情况下最大化利用计算资源的batch size经验值
BEST_TRAIN_BATCH_SIZE_MAP = {
    8: 7,
    16: 8,
    32: 9,
    64: 9,
    128: 11,
    256: 12
}


class SaveCallback(TrainerCallback):
    def __init__(self, files_to_save, args_to_save={}):
        self.files_to_save = files_to_save
        self.args_to_save = args_to_save

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = f"checkpoint-{state.global_step}"
        artifact_path = os.path.join(args.output_dir, ckpt_dir)
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


@dataclass
class Arguments(TrainingArguments):
    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lm_type: str = field(
        default='seq2seq',
        metadata={"help": "Type of language model"}
    )

    cpu_offload: bool = field(
        default=False, metadata={"help": "Whether to use cpu_offload"})

    resume_and_skip_data_if_nan: bool = field(
        default=False, metadata={"help": "Whether to resume from last checkpoint and skip some data if nan"})

    no_save_atorch_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to save deepspeed checkpoint."})
    extra_save_by_epoch: bool = field(
        default=False, metadata={"help": "Whether to save checkpoint at the end of each epoch."})
    blocking_save: bool = field(
        default=False, metadata={"help": "Whether to save checkpoint synchronously or asynchronously"})
    custom_lr_scheduler_type: str = field(
        default=None, metadata={"help": "Type of language model"}
    )

    peft_type: str = field(
        default=None, metadata={"help": "Whether use peft"})
    prompt_init_text: str = field(
        default=None, metadata={"help": "The init text of prompt learning"})

    save_policy: str = field(
        default="steps",
        metadata={
            "help": "The checkpoint save strategy to use. choices: steps, epoch, interval"},
    )

    save_interval: int = field(
        default=1800,
        metadata={
            "help": "The time interval to save checkpoint, if save_strategy is set to 'interval'"}
    )

    save_nan_checkpoint: bool = field(
        default=False, metadata={"help": "Whether to save checkpoint when nan loss occur"})

    shuffle: bool = field(
        default=False, metadata={"help": "Whether to shuffle data"})

    max_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length"}
    )
    max_input_length: int = field(
        default=1024,
        metadata={"help": "The maximum total prompt length"}
    )
    max_output_length: int = field(
        default=1024,
        metadata={"help": "The maximum total output length"}
    )
    loss_func: str = field(
        default='sample_level_cross_entropy', metadata={"help": "loss func"}
    )
    undirectional_attention: bool = field(
        default=False, metadata={"help": "undirectional attention"})
    mini_batch: int = field(
        default=2,
        metadata={
            "help": "Mini batch used when loss func is set to mini_batch_token_level_cross_entropy"}
    )
    eos_token: str = field(
        default='<|endofpiece|>', metadata={"help": "end token to use"}
    )
    train_data: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Path to test data"}
    )
    auto_batch_size: bool = field(
        default=False, metadata={"help": "Auto set the best batchsize"})
    dynamic_padding: bool = field(
        default=False, metadata={"help": "Whether dynamically padding each batch"})

    no_append_glm_mask: bool = field(
        default=False, metadata={"help": "Whether to append glm_mask"})
    glm_mask: str = field(
        default='[gMASK]', metadata={"help": "Mask to use in glm"}
    )
    gpt_data: bool = field(
        default=False, metadata={"help": "Whether train gpt_data"})

    left_truncate: bool = field(
        default=False, metadata={"help": "Whether truncate at the left side"})
    isolation_position_ids: bool = field(
        default=False, metadata={"help": "Whether isolate position ids in pack training manner"})
    add_cls: bool = field(
        default=True, metadata={"help": "Whether add [CLS] token in each pack training manner"})
    online_packed: bool = field(
        default=True, metadata={"help": "Whether use online pack data process"})
    rotary_type: str = field(
        default="none", metadata={"help": "Which rotary method used in data construct"})
    atorch_opt: str = field(
        default="fsdp", metadata={"help": "atorch training optimization strategy. Support 'fsdp' and 'ddp'."}
    )
    save_load_by_streaming: bool = field(
        default=False, metadata={"help": "Accelerate save/load speed."}
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
    bits: int = field(
        default=4, metadata={"help": "How many bits to use when using qlora. Should be 4 or 8."}
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Use gradient checkpointing to save CUDA memory."}
    )
    use_packed_data: bool = field(
        default=False, metadata={"help": "Use packed dataset for training."}
    )
    use_long_glm: bool = field(
        default=False, metadata={"help": "Use Fot dataset for training."}
    )
    long_glm_factor: int = field(
        default=2, metadata={"help": "The scale factor to expand the max data length."}
    )
    datasetv2: bool = field(
        default=False, metadata={"help": "Whether to use dataset v2"})


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (
        bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def find_best_train_batch_size(world_size):
    sorted_map = sorted(BEST_TRAIN_BATCH_SIZE_MAP.items(), key=lambda x: x[0])
    best_batch_size = 1
    for conf in sorted_map:
        if world_size < conf[0]:
            break
        best_batch_size = conf[1]
    return best_batch_size


def main():
    atorch.init_distributed(os.getenv(
        "TORCH_DISTRIBUTED_BACKEND", "nccl"), timeout=timedelta(seconds=2700))
    # atorch.init_distributed("nccl", timeout=timedelta(seconds=2700))
    local_rank = atorch.local_rank()
    torch.cuda.set_device(local_rank)
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.save_policy == 'epoch':
        args.extra_save_by_epoch = False
    set_seed(args.seed)
    logger = logging.getLogger(__name__)
    mpu.initialize_model_parallel(1)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))
    if args.resume_from_checkpoint == 'true':
        logger.info(f'Resume from {args.output_dir}')
        resume_from_checkpoint = True
    else:
        logger.info('Train from scratch')
        resume_from_checkpoint = False
    if args.auto_batch_size:
        best_train_batch_size = find_best_train_batch_size(world_size)
        args.per_device_train_batch_size = best_train_batch_size
        logger.info(f'found best train batch size: {best_train_batch_size}')

    train_data_path = args.train_data
    test_data_path = args.test_data
    lm_type = args.lm_type

    old_version_tokenizer = False
    if lm_type == 'seq2seq':
        auto_model_class = GLMModel  # noqa
        if args.use_packed_data is True:
            dataset_class = GLMPackedDataset
        elif args.use_long_glm is True:
            dataset_class = GLMFoTDataset
            args.max_length *= args.long_glm_factor
        else:
            dataset_class = GLMInstructionDataset
        auto_config_class = GLMConfig
        if is_oldest_version(args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import \
                GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            args.glm_mask = '[sMASK]'
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
        auto_config_class = GLMConfig
        auto_model_class = GLMForEmbedding
        dataset_class = GLMEmbeddingDataset
        if is_oldest_version(args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import \
                GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            args.glm_mask = '[sMASK]'
            old_version_tokenizer = True
        else:
            auto_tokenization_class = GLMTokenizer
            old_version_tokenizer = False
    else:
        auto_config_class = AutoConfig
        auto_model_class = AutoModelForCausalLM  # noqa
        dataset_class = InstructionDataset
        auto_tokenization_class = AutoTokenizer

    # logger.info('Build model', ranks=[0])
    local_rank = global_rank % torch.cuda.device_count()
    device = {'': local_rank if torch.cuda.is_available() else 'cpu'}
    logger.info(
        f'world_size: {world_size}, global_rank: {global_rank}, local_rank: {local_rank}, device: {device}')
    tokenizer = auto_tokenization_class.from_pretrained(  # noqa
        args.pretrained_model_name_or_path, trust_remote_code=True)

    if args.use_long_glm is True:
        config = auto_config_class.from_pretrained(
            args.pretrained_model_name_or_path)
        config.focused_attention = True
        config.use_cache = True
    else:
        config = None

    if args.save_load_by_streaming and not args.blocking_save:
        raise ValueError("save_load_by_streaming only support blocking_save")
    if args.save_load_by_streaming:
        streaming_ckpt_path = args.pretrained_model_name_or_path
        resume_checkpoint_dir = get_last_checkpoint(
            args.output_dir, save_load_by_streaming=args.save_load_by_streaming)
        if resume_checkpoint_dir is not None:
            streaming_ckpt_path = os.path.join(
                resume_checkpoint_dir, STREAMING_CKPT_DIR)
            if os.path.exists(streaming_ckpt_path):
                streaming_ckpt_dir = streaming_ckpt_path
            else:
                raise FileNotFoundError("streaming_ckpt_path does not exist.")
        else:
            raise RuntimeError(
                f"Cannot get last ckpt from output_dir {args.output_dir}")
        config = auto_config_class.from_pretrained(
            args.pretrained_model_name_or_path)
        if args.use_long_glm is True:
            config.focused_attention = True
            config.use_cache = True

        logger.info(
            f"Load config file from {args.pretrained_model_name_or_path}. "
            f"Load atorch fsdp flat param from {streaming_ckpt_dir}."
        )
        # 此处模型放到meta device上，不占用GPU显存，在FSDP中实际初始化
        with init_empty_weights_with_disk_offload(ckpt_path=streaming_ckpt_dir, shard_kwargs={"prefix": "glm"}):
            model = auto_model_class(config)
    else:
        if args.peft_type != "qlora":
            model = auto_model_class.from_pretrained(  # noqa
                args.pretrained_model_name_or_path, config=config, trust_remote_code=True, device_map=device)
        else:
            # If using qlora, initialize BitsAndBytesConfig
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "To use qlora, please upgrade transformers to 4.30.1 by `pip install -U transformers==4.30.1`"
                )
            if bnb is None:
                raise ImportError(
                    "To use qlora, please install bitsandbytes by `pip install -U bitsandbytes==0.39.0`")
            try:
                import accelerate  # noqa
            except ImportError:
                raise ImportError(
                    "To use qlora, please install accelerate by `pip install -U accelerate==0.20.3`")
            peft_version = version.parse(peft.__version__)
            if peft_version < version.parse("0.3.0"):
                raise RuntimeError(
                    f"Qlora needs peft>=0.3.0 but current peft version is {peft_version}")
            if args.bits not in [4, 8]:
                raise ValueError(
                    f"Qlora only support 4 bits or 8 bits but got {args.bits} bits.")
            if args.bf16:
                torch_dtype = torch.bfloat16
                compute_dtype = torch.bfloat16
            elif args.fp16:
                torch_dtype = torch.float32
                compute_dtype = torch.float16
            else:
                torch_dtype = torch.float32
                compute_dtype = torch.float32
            if auto_config_class is not None:
                glm_config = auto_config_class.from_pretrained(
                    args.pretrained_model_name_or_path)
                # HACK: To avoid out-of-memory error, do not load model concurrently
                if glm_config.num_layers == 80 and world_size >= 8:
                    if local_rank >= int(world_size / 2):
                        sleep_seconds = 2200
                        logger.info(
                            f"Before loading GLM-65B, local_rank {local_rank} will sleep {sleep_seconds}"
                            " seconds to avoid cpu out of memory."
                        )
                        time.sleep(sleep_seconds)
                    logger.info(
                        f"local_rank {local_rank} start loading GLM-65B")
            model = auto_model_class.from_pretrained(  # noqa
                args.pretrained_model_name_or_path,
                config=config,
                trust_remote_code=True,
                device_map=device,
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                torch_dtype=torch_dtype,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=args.bits == 4,
                    load_in_8bit=args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            )
            if auto_config_class is not None:
                if glm_config.num_layers == 80 and world_size >= 8:
                    logger.info(
                        f"local rank {local_rank} has passed `from_pretrained`")
                    torch.distributed.barrier()

    # Load peft model
    if args.peft_type:
        peft_task_type = TaskType.ANT_EMBEDDING if lm_type == "embedding" else TaskType.ANT_CAUSAL_LM

        if args.peft_type in ["lora", "qlora"]:
            target_modules = None
            if args.peft_type == "qlora":
                model = prepare_model_for_kbit_training(model, False)
                target_modules = find_all_linear_names(args, model)
            peft_config = LoraConfig(
                task_type=peft_task_type,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
        elif args.peft_type == "unipelt":
            peft_config = UniPELTConfig(
                task_type=peft_task_type,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout
            )
        elif args.peft_type == "ptuning":
            peft_config = PromptEncoderConfig(
                task_type=peft_task_type,
                inference_mode=False,
                encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
                encoder_num_layers=args.ptuning_encoder_num_layers,
                encoder_dropout=args.ptuning_encoder_dropout,
                encoder_hidden_size=args.encoder_hidden_size,
                num_virtual_tokens=args.num_virtual_tokens
            )
        elif args.peft_type == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=peft_task_type,
                inference_mode=False,
                num_virtual_tokens=args.num_virtual_tokens,
                num_attention_heads=model.config.num_attention_heads,
                num_layers=model.config.num_layers,
                encoder_hidden_size=args.encoder_hidden_size,
                token_dim=model.config.hidden_size
            )
        elif args.peft_type == "prompt":
            peft_config = PromptTuningConfig(
                task_type=peft_task_type,
                inference_mode=False,
                prompt_tuning_init=PromptTuningInit.TEXT if args.prompt_init_text else PromptTuningInit.RANDOM,
                num_virtual_tokens=args.num_virtual_tokens,
                prompt_tuning_init_text=args.prompt_init_text if args.prompt_init_text else None,
                tokenizer_name_or_path=args.pretrained_model_name_or_path
            )
        elif args.peft_type == "roem":
            peft_config = PeftROEMConfig(
                task_type=peft_task_type,
                inference_mode=False
            )
        elif args.peft_type == "bitfit":
            peft_config = PeftBitfitConfig(
                task_type=peft_task_type,
                inference_mode=False
            )
        elif args.peft_type == "adalora":
            peft_config = AdaLoraConfig(
                task_type=peft_task_type,
                inference_mode=False,
                lora_alpha=args.lora_alpha,
                init_r=args.adalora_target_rank,
                target_r=args.adalora_init_rank,
                lora_dropout=args.lora_dropout,
            )
            raise NotImplementedError("Not support when using atorch_trainer")
        else:
            raise ValueError(
                "The param 'peft_type' must in " +
                "['lora', 'ptuning', 'prefix', 'prompt', 'bitfit', 'roem', 'unipelt'], " +
                f"but get: {args.peft_type}")
        logger.info(
            f"Load Peft {args.peft_type} model ......")

        if isinstance(peft_config, PromptLearningConfig):
            logger.info(
                "User the prompt learning method, reduce the max length with virtual tokens.")
            args.max_length -= args.num_virtual_tokens
            args.max_input_length -= args.num_virtual_tokens // 2
            args.max_output_length -= args.num_virtual_tokens // 2

        if args.gradient_checkpointing and args.peft_type in ["lora", "qlora"]:
            # Make Lora and gradient checkpointing compatible
            # https://github.com/huggingface/peft/issues/137
            model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        logger.info(
            f"Reduce trainalbe params:\n{model.print_trainable_parameters()}")
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    files_to_save = ['merge.model', 'tokenizer_config.json']
    files_to_save = [os.path.join(
        args.pretrained_model_name_or_path, filename) for filename in files_to_save]

    if os.path.exists(os.path.join(args.pretrained_model_name_or_path, 'hyper_parameters.json')):
        pretrain_model_args = json.load(
            open(os.path.join(args.pretrained_model_name_or_path, 'hyper_parameters.json')))
        if 'eos_token' in pretrain_model_args:
            args.eos_token = pretrain_model_args['eos_token']
        if 'rotary_type' in pretrain_model_args:
            args.rotary_type = pretrain_model_args['rotary_type']

    # logger.info(f'Build data loader from path {train_data_path}', ranks=[0])
    if args.datasetv2:
        if is_yaml(train_data_path):
            train_dataset = AutoDataset.from_config(train_data_path, tokenizer)
        else:
            train_dataset = GLMSeq2SeqDataset(name="train",
                                              data_path=train_data_path,
                                              tokenizer=tokenizer,
                                              max_length=args.max_length,
                                              max_input_length=args.max_input_length,
                                              max_output_length=args.max_output_length,
                                              left_truncate=args.left_truncate,
                                              scatter_num=world_size)
        if is_yaml(test_data_path):
            test_dataset = AutoDataset.from_config(test_data_path, tokenizer)
        else:
            test_dataset = GLMSeq2SeqDataset(name="test",
                                             data_path=test_data_path,
                                             tokenizer=tokenizer,
                                             max_length=args.max_length,
                                             max_input_length=args.max_input_length,
                                             max_output_length=args.max_output_length,
                                             left_truncate=args.left_truncate,
                                             scatter_num=1)
    else:
        print(f"Building train data loader from path {train_data_path}")
        train_dataset = dataset_class(data_path=train_data_path,
                                      tokenizer=tokenizer,
                                      max_length=args.max_length,
                                      max_input_length=args.max_input_length,
                                      max_output_length=args.max_output_length,
                                      return_dict=False,
                                      no_append_glm_mask=args.no_append_glm_mask,
                                      gpt_data=args.gpt_data,
                                      left_truncate=args.left_truncate,
                                      world_size=world_size,
                                      global_rank=global_rank,
                                      shard_data=True,
                                      shuffle=args.shuffle,
                                      glm_mask=args.glm_mask,
                                      old_version_tokenizer=old_version_tokenizer,
                                      split="train",
                                      undirectional_attention=args.undirectional_attention,
                                      tmp_pack_dir=os.path.join(
                                          args.output_dir, 'packed_data/train'),
                                      eos_token=args.eos_token,
                                      isolation_position_ids=args.isolation_position_ids,
                                      add_cls=args.add_cls,
                                      online_packed=args.online_packed,
                                      rotary_type=args.rotary_type
                                      )
        print(f"Building test data loader from path {test_data_path}")
        test_dataset = dataset_class(data_path=test_data_path,
                                     tokenizer=tokenizer,
                                     max_length=args.max_length,
                                     max_input_length=args.max_input_length,
                                     max_output_length=args.max_output_length,
                                     left_truncate=args.left_truncate,
                                     return_dict=False,
                                     world_size=world_size,
                                     global_rank=global_rank,
                                     shard_data=False,
                                     shuffle=args.shuffle,
                                     no_append_glm_mask=args.no_append_glm_mask,
                                     gpt_data=args.gpt_data,
                                     glm_mask=args.glm_mask,
                                     old_version_tokenizer=old_version_tokenizer,
                                     split="test",
                                     undirectional_attention=args.undirectional_attention,
                                     tmp_pack_dir=os.path.join(
                                         args.output_dir, 'packed_data/valid'),
                                     eos_token=args.eos_token,
                                     isolation_position_ids=args.isolation_position_ids,
                                     add_cls=args.add_cls,
                                     online_packed=args.online_packed,
                                     rotary_type=args.rotary_type
                                     )

    trainer = AtorchTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        evaluator=None,
        files_to_save=files_to_save,
        args_to_save={
            'max_length': args.max_length if not args.use_long_glm else args.max_length // args.long_glm_factor,
            'peft_type': args.peft_type,
            'gpt_model': args.gpt_data,
            'eos_token': args.eos_token,
            'rotary_type': args.rotary_type,
        },
        blocking_save=args.blocking_save,
        train_peft=True if args.peft_type else False,
        custom_lr_scheduler_type=args.custom_lr_scheduler_type,
        dynamic_padding=args.dynamic_padding,
        pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token),
        no_save_atorch_checkpoint=args.no_save_atorch_checkpoint,
        rank=global_rank
    )
    if args.do_train:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == '__main__':
    main()
