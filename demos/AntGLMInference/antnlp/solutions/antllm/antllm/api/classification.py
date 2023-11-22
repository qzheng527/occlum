# coding=utf-8
# @Date: 2023-06-14
from .model import LLMModel
from ..utils.aistudio_utils import AntLLMk8sConf


class Classification(LLMModel):
    def __init__(self, model):
        super().__init__(model)
        pass

    def analyse_data(self, fpath, report_path):
        '''
        分类的数据检查作为推荐环节，并产出详细的数据分析报告。
        可以把一些异常数据（比如超长）过滤出来，并给出一些针对性的建议。
        :param fpath:
        :param report_path:
        :return:
        '''
        pass

    def train_local(self, train_fpath, output_dir, validation_fpath=None, peft=None, epoch=2):
        '''
        执行本地训练
        :param train_fpath:
        :param output_dir:
        :param validation_fpath:
        :param peft:
        :param epoch:
        :return:
        '''

    def train_remote(self, train_fpath, output_dir, k8s_conf: AntLLMk8sConf,
                     validation_fpath=None, peft=None, epoch=2):
        '''
        执行远程训练
        :param train_fpath:
        :param output_dir:
        :param k8s_conf:
        :param validation_fpath:
        :param peft:
        :param epoch:
        :return:
        '''
        pass

    def generate(self, prompt, normalizer=None):
        '''
        预测的时候会调用，可选后处理器
        :param prompt:
        :param normalizer:
        :return:
        '''
        pass
