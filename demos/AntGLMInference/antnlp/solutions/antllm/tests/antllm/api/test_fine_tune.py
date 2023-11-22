import os
import pytest # noqa
from unittest import TestCase, main
from unittest.mock import patch
from solutions.antllm.antllm.api import FineTune
from solutions.antllm.antllm.api.error import JobPrepareError


class MockPeft():
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return "test_run_id"
    

class MyTestCase(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.output_dir = "experiments/fine-tune-local/"
        self.data_file_path = os.path.join(
            self.base_dir, "../data/dataset/glm_instruction_dataset_test_data.jsonl"
        )

    # @pytest.mark.skip(reason="aci timeout and the complex dependent input files")
    def test_train_local(self):
        tuner = FineTune("solutions/antllm/glm-super-mini-model")

        flag = tuner.train_local(
            train_fpath=self.data_file_path,
            output_dir=self.output_dir,
            validation_fpath=self.data_file_path,
            epoch=1
        )
        self.assertTrue(flag)
        self.assertTrue(os.path.exists(self.output_dir))

    def test_train_local_lora(self):
        tuner = FineTune("solutions/antllm/glm-super-mini-model")

        flag = tuner.train_local(
            train_fpath=self.data_file_path,
            output_dir=self.output_dir,
            validation_fpath=self.data_file_path,
            peft="lora",
            epoch=1
        )
        self.assertTrue(flag)
        self.assertTrue(os.path.exists(self.output_dir))

    @patch('solutions.antllm.antllm.api.fine_tune.submit_aistudio_task_v2', return_value='mock-id')
    @patch('solutions.antllm.antllm.api.fine_tune.FineTune.init_remote_run', return_value='mock-id')
    def test_train_remote(self, mock_init_run, mock_submit_aistudio_task):
        from solutions.antllm.antllm.utils.aistudio_utils import AntLLMk8sConf
        mock_init_run.return_value = 'test_run_id'
        k8s_conf = AntLLMk8sConf(app_name='gbank', gpu_num=8, init_command='')
        dataset_id = 'alpaca_train_dev'
        tunner = FineTune('solutions/antllm/glm-super-mini-model')
        with self.assertRaises(JobPrepareError):
            tunner.train_remote(dataset_id=dataset_id, k8s_conf=k8s_conf)
        tunner = FineTune('AntGLM-10B-SFT-20230602')
        task_id = tunner.train_remote(dataset_id=dataset_id, k8s_conf=k8s_conf)
        self.assertEqual(task_id, 'test_run_id')

    @patch('solutions.antllm.antllm.api.fine_tune.PeftSolutionRunPredict', return_value=MockPeft())
    def test_batch_predict(self, *args):
        from solutions.antllm.antllm.utils.aistudio_utils import AntLLMk8sConf
        tunner = FineTune('AntGLM-10B-SFT-20230602')
        k8s_conf = AntLLMk8sConf(app_name='gbank', gpu_num=8, init_command='')
        task_id = tunner.batch_predict("model", "predictions", k8s_conf)
        self.assertEqual(task_id, 'test_run_id')


if __name__ == '__main__':
    main()
