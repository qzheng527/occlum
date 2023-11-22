
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from enum import Enum
from transformers.optimization import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"
    CONSTANT_WARMUP = "constant_warmup"
    COSINE_WARMUP = "cosine_warmup"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    if name == SchedulerName.LINEAR:
        return LinearLR
    if name == SchedulerName.CONSTANT_WARMUP:
        return get_constant_schedule_with_warmup
    if name == SchedulerName.COSINE_WARMUP:
        return get_cosine_schedule_with_warmup
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(f"`{name}` is not a supported scheduler. " f"Supported schedulers are: {supported_schedulers}")