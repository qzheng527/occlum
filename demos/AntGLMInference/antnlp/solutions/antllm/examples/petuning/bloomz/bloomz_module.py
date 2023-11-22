from typing import Any
from deepspeed.ops.adam import FusedAdam
from alps.pytorch.lightning import LightningModule
import torch
from transformers import (
    get_linear_schedule_with_warmup,
)
from alps.util import logger


class BloomzModule(LightningModule):
    def __init__(
        self,
        tokenizer,
        model,
        model_input_field="question",
        model_output_field="answer",
        extra_fields: list = [],
        generate_params: dict = {
            "repetition_penalty": 1.0,
            "do_sample": False,
            "temperature": 1.3,
            "top_p": 0.6,
            "max_length": 800,
        },
        learning_rate: float = 5e-6,
        adam_epsilon: float = 3e-8,
        warmup_steps: int = 30,
        weight_decay: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        logger.info("alps module hparams:{}".format(self.hparams))
        self.tokenizer = tokenizer

        self.model = model

        # generate参数
        self.generate_params = generate_params

        # io 相关
        self.model_input_field = model_input_field
        self.model_output_field = model_output_field
        self.extra_fields = extra_fields

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # 自定义forward逻辑

        if labels is not None:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # token_type_ids=token_type_ids,
                labels=labels,
            )
        else:
            # 离线打分时无labels属性

            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

    def training_step(self, batch, batch_idx):
        # 打印训练数据查看数据分片是否生效
        # if batch_idx == 0:
        #     logger.info(f"train pid: {os.getpid()} batch: {batch}")

        # 定义train逻辑
        y = batch["input_ids"].squeeze(1)
        lm_labels = y.clone().detach()
        lm_labels[y == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1),
            # token_type_ids=batch['token_type_ids'],
            labels=lm_labels,
        )
        loss = outputs[0]
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_ids = batch["input_ids"].squeeze(1)
        attention_mask = batch["attention_mask"].squeeze(1)
        model_output = {}
        for k in self.extra_fields:
            # 把附加列先detach和写进输出结果
            model_output[k] = batch[k].detach().cpu().numpy().tolist()
        preds = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **self.generate_params,
        )
        # 解码
        try:
            model_output[self.model_output_field] = self.tokenizer.batch_decode(
                preds.detach().cpu().numpy().tolist()
            )
            model_output[self.model_input_field] = self.tokenizer.batch_decode(
                input_ids.squeeze(1).detach().cpu().numpy().tolist()
            )
        except Exception as e:
            logger.error("预测出错")
            logger.error(e)
            logger.error(f"preds:{preds}")
            model_output = ["error"]
        # 返回预测结果

        return model_output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # 定义validation逻辑
        y = batch["input_ids"].squeeze(1)
        lm_labels = y.clone().detach()
        lm_labels[y == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1),
            # token_type_ids=batch['token_type_ids'],
            labels=lm_labels,
        )
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=2)
        labels = batch["input_ids"].squeeze(1)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.trainer.datamodule.train_batch_size * max(
            1, self.trainer.num_devices
        )
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = FusedAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
