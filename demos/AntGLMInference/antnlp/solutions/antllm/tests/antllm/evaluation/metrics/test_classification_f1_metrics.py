from unittest import TestCase, main
import os
import solutions.antllm.tests.antllm.evaluation.metrics.data_loader as data_loader
from solutions.antllm.antllm.evaluation.metrics.classification.f1_metrics import HuggingfaceF1


class TestMetric(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.data_file = os.path.join(self.base_dir, '../data/english_eval_classification_data.txt')
        self.predictions, self.references, self.extras = data_loader.load_data(self.data_file)

    def test_metric(self):
        test = HuggingfaceF1()
        self.assertEqual(test.compute(self.predictions, self.references, self.extras), 66.67)


if __name__ == '__main__':
    main()