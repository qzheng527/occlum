import os
import json
from typing import Dict, Any
from transformers import PretrainedConfig


class FineTuneConfig(PretrainedConfig):
    def __init__(
        self,
        training_config_path: str = None,
        deepspeed_config_path: str = None,
        peft_config_path: str = None,
        local_training_port: int = 12346,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if not training_config_path or not os.path.exists(training_config_path):
            training_config_path = os.path.join(
                os.path.dirname(__file__), "fine_tune.json")
        if not deepspeed_config_path or not os.path.exists(deepspeed_config_path):
            deepspeed_config_path = os.path.join(
                os.path.dirname(__file__), "deepspeed.json")
        if not peft_config_path or not os.path.exists(peft_config_path):
            peft_config_path = os.path.join(
                os.path.dirname(__file__), "peft_config.json")
        
        self.training_config_path = training_config_path
        self.deepspeed_config_path = deepspeed_config_path
        self.peft_config_path = peft_config_path

        self.training_config = self.load_config(training_config_path)
        self.deepspeed_config = self.load_config(deepspeed_config_path)
        self.peft_config = self.load_config(peft_config_path)

        self.local_training_port = local_training_port

    @classmethod
    def load_config(self, model_name_or_path: str) -> Dict[str, Any]:
        if not os.path.exists(model_name_or_path):
            raise FileNotFoundError(f"The config file: {model_name_or_path}, not exists.")

        with open(model_name_or_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        return config
    
    def save_config(self, output_dir: str) -> None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        deepspeed_config_path = os.path.join(output_dir, "deepspeed.json")
        training_config_path = os.path.join(output_dir, "fine_tune.json")
        peft_config_path = os.path.join(output_dir, "peft_config.json")

        with open(deepspeed_config_path, "w", encoding="utf-8") as f:
            json.dump(self.deepspeed_config, f, ensure_ascii=False, indent=4)

        with open(training_config_path, "w", encoding="utf-8") as f:
            json.dump(self.training_config, f, ensure_ascii=False, indent=4)

        with open(peft_config_path, "w", encoding="utf-8") as f:
            json.dump(self.peft_config, f, ensure_ascii=False, indent=4)
