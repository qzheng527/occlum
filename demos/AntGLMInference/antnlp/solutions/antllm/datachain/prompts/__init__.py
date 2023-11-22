import os
from pathlib import Path

from solutions.antllm.datachain.utils import get_all_files

from .base import Instruct

__all__ = [
    "Instruct"
]

PROMPT_ROOT = str(Path(__file__).parent / "resources")
# relative path in resources/prompts
ALL_PROMPTS = {

}


def get_prompt_by_name(name):
    if name in ALL_PROMPTS:
        prompt_path = os.path.join(PROMPT_ROOT, ALL_PROMPTS[name])
        return open(prompt_path, "r").read()

    paths = get_all_files(PROMPT_ROOT)
    for path in paths:
        if Path(path).stem.strip().lower() == name.strip().lower():
            return open(path, "r").read()
    return ""
