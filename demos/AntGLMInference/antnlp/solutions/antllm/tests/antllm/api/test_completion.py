from unittest.mock import patch
import unittest
from solutions.antllm.antllm.api import Completion
from solutions.antllm.antllm.api import RemoteCompletion
from solutions.antllm.antllm.inference.glm_predictor import GenFinishReason


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.completer = Completion("solutions/antllm/glm-super-mini-model")

    def test_gen1(self):
        answers = self.completer.generate(
            '北京是中国首都吗？', num_return_sequences=1, num_beams=1, max_tokens=2)
        self.assertEqual(len(answers.texts), 1)
        self.assertEqual(len(answers.finish_reasons), 1)

    def test_gen1_1(self):
        answers = self.completer.generate(
            '北京是中国首都吗？', num_return_sequences=1, num_beams=1, max_tokens=1)
        self.assertEqual(len(answers.texts), 1)
        self.assertEqual(answers.finish_reasons[0], GenFinishReason.MAX_LEN)

    def test_gen1_2(self):
        answers = self.completer.generate(
            '北京是中国首都吗？', num_return_sequences=2, num_beams=2, max_tokens=1)
        self.assertEqual(len(answers.texts), 2)
        self.assertEqual(len(answers.finish_reasons), 2)

    def test_gen_batch(self):
        answers = self.completer.generate_batch(['北京是中国首都吗？', '中国西部之都在哪'],
                                                num_return_sequences=1, num_beams=1, max_tokens=1)
        self.assertEqual(len(answers), 2)
        self.assertEqual(len(answers[0].texts), 1)
        self.assertEqual(answers[0].finish_reasons[0], GenFinishReason.MAX_LEN)

    def test_gen_batch2(self):
        answers = self.completer.generate_batch(['北京是中国首都吗？', '中国西部之都在哪'],
                                                num_return_sequences=2, num_beams=2, max_tokens=2)
        self.assertEqual(len(answers), 2)
        self.assertEqual(len(answers[0].texts), 2)
        self.assertEqual(len(answers[1].texts), 2)
        self.assertEqual(len(answers[0].finish_reasons), 2)
        self.assertEqual(len(answers[1].finish_reasons), 2)

    def test_gen_stream(self):
        tokens = []
        for token in self.completer.generate_stream('北京是中国首都吗？'):
            tokens.append(token)
            print(token)
        self.assertGreater(len(tokens), 0)

    def test_gen_stream_1(self):
        tokens = []
        for token in self.completer.generate_stream('北京是中国首都吗？', max_tokens=1):
            tokens.append(token)
            print(token)
        self.assertLess(len(tokens), 2)

    def test_gen_stream_eq(self):
        tokens = []
        prompt = '北京是中国首都吗？'
        ans1 = self.completer.generate(
            prompt, num_return_sequences=1, num_beams=1, do_sample=False, max_tokens=5)
        for token in self.completer.generate_stream(prompt, max_tokens=5):
            tokens.append(token)
        ans2 = ''.join(tokens)
        self.assertEqual(ans1.texts[0].strip(), ans2.strip())


class TestRemoteCompletion(unittest.TestCase):
    def setUp(self):
        scene_name = "xuesun"
        chain_name = "v1"
        self.remote_completer = RemoteCompletion(
            scene_name=scene_name, chain_name=chain_name)
        # mock generate_maya_response
        self.generate_maya_mock_response = [
            {'texts': [',适合出门散步。我打算去公园，呼吸新鲜空气，放松身心。'], 'finish_reasons': ['EOS']}]

        # # mock generate_maya_response
        self.batch_generate_maya_mock_response = [{'texts': [',适合出门散步。我打算去公园，呼吸新鲜空气，放松身心。'], 'finish_reasons': [
            'EOS']}, {'texts': [' 很抱歉，这个问题可能会涉及到个人政治观点和立场，回答可能会引起不必要的争议或者歧视。'], 'finish_reasons': ['EOS']}]

    def test_remote_answer_without_adapter_answer(self):
        query = "今天天气不错"
        with patch('solutions.antllm.antllm.inference.remote_predictor.RemoteInference._maya_infer_client',
                   return_value=self.generate_maya_mock_response):
            completion_output = self.remote_completer.generate(
                query)

            self.assertTrue(type(completion_output.texts) == list)
            self.assertTrue(type(completion_output.finish_reasons) == list)

    def test_remote_anwers_with_adapter_path(self):
        query = "今天天气不错"
        with patch('solutions.antllm.antllm.inference.remote_predictor.RemoteInference._maya_infer_client',
                   return_value=self.generate_maya_mock_response):
            completion_output = self.remote_completer.generate(
                query, adapter_name="test")

            self.assertTrue(type(completion_output.texts) == list)
            self.assertTrue(type(completion_output.finish_reasons) == list)

    def test_remote_batch_answer_without_adapter(self):
        query = ["今天天气不错", "中国在哪里？"]
        with patch('solutions.antllm.antllm.inference.remote_predictor.RemoteInference._maya_infer_client',
                   return_value=self.batch_generate_maya_mock_response):

            batch_comletion_output = self.remote_completer.generate_batch(
                query)
            self.assertTrue(type(batch_comletion_output) == list)
            self.assertTrue(type(batch_comletion_output[0].texts) == list)
            self.assertTrue(
                type(batch_comletion_output[0].finish_reasons) == list)

    def test_remote_batch_answer_with_adapter(self):
        query = ["今天天气不错", "中国在哪里？"]
        with patch('solutions.antllm.antllm.inference.remote_predictor.RemoteInference._maya_infer_client',
                   return_value=self.batch_generate_maya_mock_response):

            batch_comletion_output = self.remote_completer.generate_batch(
                query, adapter_name="test")
            self.assertTrue(type(batch_comletion_output) == list)
            self.assertTrue(type(batch_comletion_output[0].texts) == list)
            self.assertTrue(
                type(batch_comletion_output[0].finish_reasons) == list)


if __name__ == '__main__':
    unittest.main()