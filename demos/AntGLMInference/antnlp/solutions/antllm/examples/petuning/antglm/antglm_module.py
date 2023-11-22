import torch
from alps.pytorch.lightning import LightningModule
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from alps.util import logger
import time


class AntGLMModule(LightningModule):
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
        """为antglm适配的lightningmodule，主要改了前馈、训练、预测等"""
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.tokenizer = tokenizer

        self.model = model

        # generate参数
        self.generate_params = generate_params

        # io 相关
        self.model_input_field = model_input_field
        self.model_output_field = model_output_field
        self.extra_fields = extra_fields

    def glm_post_process(self, pred):
        pass

    def forward(
        self,
        input_ids,
        generation_attention_mask,
        position_ids,
        **kwargs,
    ):
        """forward
        按照lightning的建议，只实现infer相关
        Args:
            input_ids (_type_): _description_

            attention_mask (_type_): _description_

            position_ids (_type_): _description_

            kwargs: 

        Returns:
            _type_: _description_
        """
        preds = self.model.generate(
            input_ids=input_ids,
            generation_attention_mask=generation_attention_mask,
            position_ids=position_ids,
            eos_token_id=self.tokenizer.eop_token_id,
            **kwargs,
        )
        return preds

    def training_step(self, batch, batch_idx):
        lm_labels = batch["labels"]

        outputs = self.model(
            input_ids=batch["input_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(
                1
            ),  # 见tokenizer_glm.py:296，有label（训练）时，key是attention_mask,否则是generation_attention_mask
            position_ids=batch["position_ids"].squeeze(1),
            labels=lm_labels,
        )
        loss = outputs[0]
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        model_output = {}
        # 把附加列先detach和写进输出结果
        for k in self.extra_fields:
            model_output[k] = batch[k].detach().cpu().numpy().tolist()
            # model_output[k] = batch[k].squeeze(1).detach().cpu().numpy().tolist()

        input_ids = batch["input_ids"].squeeze(1)
        # 见tokenizer_glm.py:296，有label（训练）时，key是attention_mask,否则是generation_attention_mask
        generation_attention_mask = batch["generation_attention_mask"].squeeze(1)
        position_ids = batch["position_ids"].squeeze(1)

        start = time.process_time()

        # 调用module的forward
        preds = self(
            input_ids=input_ids,
            generation_attention_mask=generation_attention_mask,
            position_ids=position_ids,
            **self.generate_params,
        )
        infer_time = time.process_time()
        logger.info(f"generation time {infer_time-start}")
        # 解码
        try:
            model_output[self.model_output_field] = self.tokenizer.batch_decode(
                preds.detach().cpu().numpy().tolist()
            )
            model_output[self.model_input_field] = self.tokenizer.batch_decode(
                batch["input_ids"].squeeze(1).detach().cpu().numpy().tolist()
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
        # y = batch['input_ids']
        lm_labels = batch["labels"]
        # lm_labels[y == self.tokenizer.eop_token_id] = -100
        outputs = self(
            input_ids=batch["input_ids"].squeeze(1),
            attention_mask=batch["attention_mask"].squeeze(1),
            position_ids=batch["position_ids"].squeeze(1),
            labels=lm_labels,
        )
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=2)
        labels = batch["labels"]

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
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                # 根据p.requires_grad判断参数是否需要更新
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                # 根据p.requires_grad判断参数是否需要更新
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        # optimizer = FusedAdam(
        #     self.parameters(),
        #     lr=self.hparams.learning_rate,
        #     eps=self.hparams.adam_epsilon,
        # )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
