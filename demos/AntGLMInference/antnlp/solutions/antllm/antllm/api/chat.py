# coding=utf-8
# @Date: 2023-06-14
from .model import LLMModel


class Chat(LLMModel):
    def __init__(self, model):
        super().__init__(model)

    def chat(self, prompt, max_tokens=32, stream=False, num_beams=5, temperature=1, topk=10, topp=0.9):
        pass

    def init_chat(self):
        '''
        多轮对话，需要先初始化，用于清空对话历史
        :return:
        '''
        pass

    def get_history_messages(self):
        '''
        获取对话历史
        :return:
        '''
        pass
