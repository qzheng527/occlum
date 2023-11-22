import pytest # noqa
from unittest import TestCase, main
from unittest.mock import patch
import solutions.antllm.antllm.utils.benchmark as benchmark


class MyTestCase(TestCase):
    def setUp(self):
        pass
    
    @patch("solutions.antllm.antllm.utils.benchmark.post_request", return_value={"message": "success"})
    def test_submit_aistudio_tas_v2(self, *args):
        import solutions.antllm.antllm.utils.aistudio_utils as aistudio_utils
        train_args = {}
        k8s_conf = aistudio_utils.AntLLMk8sConf("test", 2)
        ret = benchmark.submit_aistudio_task_v2(train_args, k8s_conf, "")
        self.assertEqual(ret, "success")

    @patch("solutions.antllm.antllm.utils.benchmark.get_request", return_value=True)
    def test_notify_benchmark_server(self, *args):
        ret = benchmark.notify_benchmark_server("", "", "", "")
        self.assertTrue(ret)


if __name__ == '__main__':
    main()