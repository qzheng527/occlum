from colossalai.amp import AMP_TYPE
from colossalai.nn.optimizer import HybridAdam

BATCH_SIZE = 1
NUM_EPOCHS = 10
MAX_INPUT_LENGTH = 500
MAX_OUTPUT_LENGTH = 500
SEQ_LEN = 1024
NUM_MICRO_BATCHES = 4
HIDDEN_SIZE = 768
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

TRAIN_DATA = "data/text_extraction_train.jsonl"
TEST_DATA = "data/text_extraction_test.jsonl"
PRETRAINED_MODEL_NAME_OR_PATH = "/workspace/chatgpt/pretrained_models/glm-10b-ant"
SAVE_CHECKPOINT_INTERVAL = 1
SAVE_CHECKPOINT_BY_ITER = False

# if you do no want zero, just comment out this dictionary
# zero = dict(model_config=dict(tensor_placement_policy='cuda', shard_strategy=TensorShardStrategy()),
#             optimizer_config=dict(initial_scale=2**5))


fp16 = dict(
    mode=AMP_TYPE.NAIVE
)

optimizer = dict(
    type=HybridAdam,
    lr=0.000015,
    weight_decay=1e-2,
)

# pipeline parallel: modify integer value for the number of pipeline stages
# tensor parallel: modify size to set the tensor parallel size, usually the number of GPUs per node
# for the current model implementation, mode can only be 1D or None

# parallel = dict(
#     pipeline=1,
# )

parallel = dict(
    pipeline=1,
    # tensor=dict(size=2, mode='1d')
)
