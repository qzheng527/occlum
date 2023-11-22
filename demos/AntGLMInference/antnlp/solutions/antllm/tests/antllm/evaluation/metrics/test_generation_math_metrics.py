from unittest import TestCase, main
import os
import solutions.antllm.tests.antllm.evaluation.metrics.data_loader as data_loader
from solutions.antllm.antllm.evaluation.metrics.generation.math_metrics import GSM8kMetric


class TestMetric(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.data_file = os.path.join(self.base_dir, '../data/math_eval_data.txt')
        self.predictions, self.references, self.extras = data_loader.load_data(self.data_file)
        print(f"{self.predictions}, {self.references}")

    def test_metric(self):
        test = GSM8kMetric()
        self.assertEqual(test.compute(self.predictions, self.references, self.extras), 100.0)


if __name__ == '__main__':
    main()