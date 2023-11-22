#!/usr/bin/env python
# coding=utf-8
# @Author: liangzhuang.mlz
# @Date: 2023-6-24

import logging

import torch
import torch.nn.functional as F
from torch import nn

from solutions.antllm.antllm.inference.glm_predictor import GLMForInference
from solutions.antllm.antllm.models.peft.modeling_peft import AntPeftForCausalLM, PeftModel  # noqa
from solutions.antllm.antllm.training.trainer.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)


class SFTDistillTrainer(SFTTrainer):
    def __init__(self,
                 *args,
                 teacher_model_path=None,
                 logit_weight=1,
                 hidden_state_cos_weight=0,
                 hidden_state_mse_weight=0,
                 hidden_states_mes_mapping='',
                 temperature=2,
                 hard_target_weight=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)

        try:
            gpu_index = int(str(self.args.device).split(':')[-1])
        except Exception:
            gpu_index = int(str(self.model.device)[-1])
        self.teacher_model = GLMForInference(teacher_model_path, gpu_index=gpu_index).model.eval().half()
        self.gpu_index = gpu_index

        self.logit_weight = logit_weight
        self.hidden_state_cos_weight = hidden_state_cos_weight
        self.hidden_state_mse_weight = hidden_state_mse_weight
        self.hidden_states_mes_mapping = []
        if hidden_states_mes_mapping is not None and hidden_states_mes_mapping != '' and \
                self.hidden_state_mse_weight > 0:
            self.hidden_states_mes_mapping = [
                (int(layer_match.strip().split(':')[0]), int(layer_match.strip().split(':')[1]))
                for layer_match in hidden_states_mes_mapping.split(",")]
        self.temperature = temperature
        self.debut_cnt = 0
        self.hard_target_weight = hard_target_weight

    def logit_loss_func(self, feature, soft_feat, temperature):
        logit_std = F.log_softmax(feature / temperature, dim=-1)
        soft_std = F.softmax(soft_feat / temperature, dim=-1)
        loss = nn.KLDivLoss(reduction='batchmean')(logit_std, soft_std) * pow(temperature, 2.0)
        return loss

    def hidden_state_cos_loss_func(self, hidden_state, teacher_state):
        loss = 1 - torch.mean(
            F.cosine_similarity(hidden_state.contiguous().float(), teacher_state.contiguous().float(), dim=-1))
        return loss

    def match_hidden_states_mse_loss_func(self, mems, teacher_mems):
        loss_mse_func = torch.nn.MSELoss()
        loss = torch.FloatTensor([0])
        for (t_idx, s_idx) in self.hidden_states_mes_mapping:
            loss += loss_mse_func(mems[s_idx], teacher_mems[t_idx])
        loss = loss / len(self.hidden_states_mes_mapping)
        return loss

    def hidden_states_mse_loss_func(self, hidden_state, teacher_state):
        loss = torch.nn.MSELoss()(hidden_state, teacher_state)
        return loss

    def compute_loss(self, model, inputs, return_outputs=False):
        self.debut_cnt += 1
        # 从transformers里面copy过来后修改的，用于计算
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        loss_hard = outputs.loss

        loss = loss_hard * self.hard_target_weight

        # teacher info
        with torch.no_grad():
            teacher_outs = self.teacher_model(**inputs)

        # valid labels pos
        labels = inputs.pop("labels").view(-1)
        valid_list = []
        for i in range(labels.size()[0]):
            if labels[i].item() >= 0:
                valid_list.append(i)

        # logit loss
        if self.logit_weight > 0:
            logit_student = outputs.logits.view(-1, outputs.logits.size(-1))[valid_list, :]
            logit_teacher = teacher_outs.logits.view(-1, teacher_outs.logits.size(-1))[valid_list, :]
            loss_logit = self.logit_loss_func(logit_student, logit_teacher, self.temperature) * self.logit_weight
            loss += loss_logit
            loss_logit_print = loss_logit.item()
        else:
            loss_logit_print = 0

        # 对最后一层hidden_states，
        if self.hidden_state_cos_weight > 0 or self.hidden_state_mse_weight > 0:
            student_hidden_state = outputs.last_hidden_states.view(
                -1, outputs.last_hidden_states.size(-1))[valid_list, :]
            teacher_hidden_state = teacher_outs.last_hidden_states.view(
                -1, teacher_outs.last_hidden_states.size(-1))[valid_list, :]

            # 做 F.cosine_similarity
            if self.hidden_state_cos_weight > 0:
                loss_hidden_state_cos = self.hidden_state_cos_loss_func(
                    student_hidden_state, teacher_hidden_state) * self.hidden_state_cos_weight
                loss += loss_hidden_state_cos
                loss_hidden_state_cos_print = loss_hidden_state_cos.item()
            else:
                loss_hidden_state_cos_print = 0

            # 对map里面的layer的hidden_states，做 mse
            if self.hidden_state_mse_weight > 0:
                loss_hidden_state_mse = self.hidden_states_mse_loss_func(
                    student_hidden_state, teacher_hidden_state) * self.hidden_state_mse_weight
                loss += loss_hidden_state_mse
                loss_hidden_state_mse_print = loss_hidden_state_mse.item()
            else:
                loss_hidden_state_mse_print = 0
        else:
            loss_hidden_state_cos_print = 0
            loss_hidden_state_mse_print = 0

        # 每100条，打印loss
        if self.debut_cnt % 100 == 1 and self.gpu_index == 0:
            print('loss_hard, teacher, logit, hidden_cos, hidden_mse', loss_hard.item(), teacher_outs.loss.item(),
                  loss_logit_print, loss_hidden_state_cos_print, loss_hidden_state_mse_print)

        outputs.loss = loss
        return (loss, outputs) if return_outputs else loss
