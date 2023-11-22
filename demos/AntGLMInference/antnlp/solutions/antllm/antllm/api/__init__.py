# coding=utf-8
# @Author: jianiu.lj
# @Date: 2023-06-13

from .chat import Chat
from .classification import Classification
from .completion import Completion, RemoteCompletion
from .embedding import Embedding
from .fine_tune import FineTune
from ..utils.aistudio_utils import AntLLMk8sConf
from .distill import Distill
from .define import VERSION


__version__ = VERSION


__all__ = [
    "__version__",
    "Chat",
    "Classification",
    "Completion",
    "RemoteCompletion",
    "Embedding",
    "FineTune",
    "Distill"
]
