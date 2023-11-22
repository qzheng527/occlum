import logging
import os
from pathlib import Path
import torch

from dataclasses import dataclass, field
from peft import (  # noqa
    LoraConfig,
    PromptLearningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptEncoderReparameterizationType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model
)
from solutions.antllm.antllm.data.dataset.chatglm_instruction_dataset import (
    ChatGLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.glm_instruction_dataset import (
    GLMInstructionDataset
)
from solutions.antllm.antllm.data.dataset.instruction_dataset import (
    InstructionDataset
)
from solutions.antllm.antllm.data.dataset.glm_embedding_dataset import (
    GLMEmbeddingDataset
)
from solutions.antllm.antllm.models.chatglm.modeling_chatglm import (
    ChatGLMForConditionalGeneration
)
from solutions.antllm.antllm.models.chatglm.tokenization_chatglm import (
    ChatGLMTokenizer
)
from solutions.antllm.antllm.models.glm.modeling_glm import (
    GLMForConditionalGeneration
)
from solutions.antllm.antllm.models.embeddings.modeling_embedding import (
    GLMForEmbedding
)
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
from solutions.antllm.antllm.models.peft.tuner import (
    AdaLoraConfig,
    PeftROEMConfig,
    PeftBitfitConfig,
    UniPELTConfig
)
from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM
from solutions.antllm.antllm.training.trainer.sft_distill_trainer import SFTDistillTrainer
from solutions.antllm.antllm.utils.version_utils import is_oldest_version
from transformers import (  # noqa
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

from solutions.antllm.antllm.commands.sft.train_deepspeed import (
    DynamicPaddingCollator,
    SaveCallback,
    EpochSaveCallback,
    PeftArguments,
    DataArguments,
    ModelArguments
)


@dataclass
class DistillArguments:
    teacher_model_path: str = field(
        metadata={
            "help": "Path to teacher model or model identifier from huggingface.co/models"}
    )
    logit_weight: float = field(
        default=1.0, metadata={"help": "distill logit loss weight, default 1.0"}
    )
    hard_target_weight: float = field(
        default=1.0, metadata={"help": "hard_target loss weight, default 1.0"}
    )
    hidden_state_cos_weight: float = field(
        default=0.0, metadata={"help": "last hidden_state cos loss weight, default 0.0"}
    )
    hidden_state_mse_weight: float = field(
        default=0.0, metadata={"help": "hard_target mes loss weight, default 0.0"}
    )
    hidden_states_mes_mapping: str = field(
        default='', metadata={"help": "hard_target mes loss layer_mapping, default ''"}
    )
    temperature: float = field(
        default=2.0, metadata={"help": "distill temperature, default 2.0"}
    )


def load_model_by_type(lm_type, model_args, data_args, global_rank):
    old_version_tokenizer = False
    if lm_type == 'seq2seq':
        auto_model_class = GLMForConditionalGeneration  # noqa
        dataset_class = GLMInstructionDataset
        if is_oldest_version(model_args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            data_args.glm_mask = '[sMASK]'
            old_version_tokenizer = True
        else:
            auto_tokenization_class = GLMTokenizer
            old_version_tokenizer = False
    elif lm_type == "chatglm":
        auto_model_class = ChatGLMForConditionalGeneration
        dataset_class = ChatGLMInstructionDataset
        auto_tokenization_class = ChatGLMTokenizer
    elif lm_type == "embedding":
        auto_model_class = GLMForEmbedding
        dataset_class = GLMEmbeddingDataset
        if is_oldest_version(model_args.pretrained_model_name_or_path):
            from solutions.antllm.antllm.models.glm.tokenization_glm_deprecated import GLMChineseTokenizer
            auto_tokenization_class = GLMChineseTokenizer
            data_args.glm_mask = '[sMASK]'
            old_version_tokenizer = True
        else:
            auto_tokenization_class = GLMTokenizer
            old_version_tokenizer = False
    else:
        auto_model_class = AutoModelForCausalLM  # noqa
        dataset_class = InstructionDataset
        auto_tokenization_class = AutoTokenizer

    # logger.info('Build model', ranks=[0])
    local_rank = global_rank % torch.cuda.device_count()
    device = {'': local_rank if torch.cuda.is_available() else 'cpu'}
    tokenizer = auto_tokenization_class.from_pretrained(  # noqa
        model_args.pretrained_model_name_or_path, trust_remote_code=True)

    model = auto_model_class.from_pretrained(  # noqa
        model_args.pretrained_model_name_or_path, trust_remote_code=True, device_map=device)

    return tokenizer, model, dataset_class, old_version_tokenizer


def try_load_peft_model(model, lm_type, model_args, training_args, peft_args, data_args, resume_from_checkpoint,
                        logger):
    peft_task_type = TaskType.ANT_EMBEDDING if lm_type == "embedding" else TaskType.ANT_CAUSAL_LM
    num_virtual_tokens = 20

    if peft_args.peft_type == "lora":
        peft_config = LoraConfig(
            task_type=peft_task_type,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
    elif peft_args.peft_type == "unipelt":
        peft_config = UniPELTConfig(
            task_type=peft_task_type,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
    elif peft_args.peft_type == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=peft_task_type,
            inference_mode=False,
            encoder_reparameterization_type=PromptEncoderReparameterizationType.MLP,
            encoder_num_layers=2,
            encoder_dropout=0.1,
            encoder_hidden_size=128,
            num_virtual_tokens=num_virtual_tokens
        )
    elif peft_args.peft_type == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=peft_task_type,
            inference_mode=False,
            num_virtual_tokens=num_virtual_tokens,
            prefix_projection=True,
            num_attention_heads=model.config.num_attention_heads,
            num_layers=model.config.num_layers,
            encoder_hidden_size=128,
            token_dim=model.config.hidden_size
        )
    elif peft_args.peft_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type=peft_task_type,
            inference_mode=False,
            prompt_tuning_init=PromptTuningInit.TEXT if peft_args.prompt_init_text else PromptTuningInit.RANDOM,
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init_text=peft_args.prompt_init_text,
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
            lora_alpha=32,
            init_r=40,
            target_r=32,
            lora_dropout=0.1,
        )
    else:
        raise ValueError(
            f"The param 'peft_type' must in " +
            f"['lora', 'ptuning', 'prefix', 'prompt', 'adalora', 'unipelt', 'roem', 'bitfit'], " +
            f"but get: {peft_args.peft_type}")
    logger.info(
        f"Load Peft {peft_args.peft_type} model ......")

    if isinstance(peft_config, PromptLearningConfig):
        logger.info(
            f"User the prompt learning method, reduce the max length with virtual tokens.")
        data_args.max_length -= num_virtual_tokens
        data_args.max_input_length -= num_virtual_tokens // 2
        data_args.max_output_length -= num_virtual_tokens // 2

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
    logger.info(
        f"Reduce trainalbe params:\n{model.print_trainable_parameters()}")
    return model


def main():
    # parser param
    # 与train_deepspeed的差异，主要是 获取distill_args的参数
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PeftArguments, DistillArguments))
    model_args, data_args, training_args, peft_args, distill_args = parser.parse_args_into_dataclasses()
    logger = logging.getLogger(__name__)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))
    podname = os.environ.get('ILOGTAIL_PODNAME', 'master')

    if 'master' in podname:
        save_checkpoint = True
    elif 'worker' in podname:
        save_checkpoint = False
    else:
        save_checkpoint = True
    if training_args.resume_from_checkpoint == 'true':
        logger.info(f'Resume from {training_args.output_dir}')
        resume_from_checkpoint = True
    else:
        logger.info(f'Train from scratch')
        resume_from_checkpoint = False
    logger.info(f'world_size: {world_size}, global_rank: {global_rank}')

    train_data_path = data_args.train_data
    test_data_path = data_args.test_data
    lm_type = model_args.lm_type

    # Load model by type and args
    tokenizer, model, dataset_class, old_version_tokenizer = load_model_by_type(
        lm_type, model_args, data_args, global_rank)

    # Load peft model
    if peft_args.peft_type:
        model = try_load_peft_model(model, lm_type, model_args, training_args, peft_args, data_args,
                                    resume_from_checkpoint, logger)

    files_to_save = []
    for filename in Path(model_args.pretrained_model_name_or_path).iterdir():
        filename = str(filename)
        if filename.endswith('pytorch_model.bin') \
                or filename.endswith('hyper_parameters.json') \
                or filename.endswith('adapter_config.json') \
                or filename.endswith('adapter_model.bin'):
            continue
        files_to_save.append(filename)

    # logger.info(f'Build data loader from path {train_data_path}', ranks=[0])
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
                                  old_version_tokenizer=old_version_tokenizer
                                  )
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
                                 old_version_tokenizer=old_version_tokenizer
                                 )
    data_collator = None
    if data_args.dynamic_padding:
        data_collator = DynamicPaddingCollator(
            pad_id=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))

    # 与train_deepspeed的差异，主要是 使用了SFTDistillTrainer 以及 distill_args的参数
    trainer = SFTDistillTrainer(model=model,
                                args=training_args,
                                train_dataset=train_dataset,
                                eval_dataset=test_dataset,
                                data_collator=data_collator,
                                callbacks=[SaveCallback(files_to_save, args_to_save={
                                    'max_length': data_args.max_length,
                                    'peft_type': peft_args.peft_type,
                                    'gpt_model': data_args.gpt_data})],
                                no_save_deepspeed_checkpoint=model_args.no_save_deepspeed_checkpoint,
                                save_pytorch_model_bin_checkpoint=save_checkpoint, rank=global_rank,
                                train_peft=True if peft_args.peft_type else False,
                                no_save_base_model=model_args.no_save_base_model,

                                teacher_model_path=distill_args.teacher_model_path,
                                logit_weight=distill_args.logit_weight,
                                hard_target_weight=distill_args.hard_target_weight,
                                hidden_state_cos_weight=distill_args.hidden_state_cos_weight,
                                hidden_state_mse_weight=distill_args.hidden_state_mse_weight,
                                hidden_states_mes_mapping=distill_args.hidden_states_mes_mapping,
                                temperature=distill_args.temperature
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

    if training_args.do_eval:
        trainer.evaluate()


if __name__ == '__main__':
    main()
