from unittest import TestCase, main
import os
import solutions.antllm.tests.antllm.evaluation.metrics.data_loader as data_loader
from solutions.antllm.antllm.evaluation.metrics.perplexity_metrics import Perplexity


class TestMetric(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.data_file = os.path.join(self.base_dir, '../data/perplexity_eval_data.txt')
        self.predictions, self.references, self.extras = data_loader.load_data(self.data_file)

    def test_metric(self):
        test = Perplexity()
        self.assertEqual(round(test.compute(self.predictions, self.references, self.extras), 2), 5.30)


if __name__ == '__main__':
    main()