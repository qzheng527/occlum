import unittest
import torch

from solutions.antllm.antllm.api import Embedding


class MyTestCase(unittest.TestCase):
    def test_embedding(self):
        texts = [
            "我觉得这只猫长得可爱",
            "这条狗好丑",
            "这只猫长得真可爱",
            "请问我的支付宝账号如何注销",
            "请问我如何注销我的支付宝，我不想用了",
            "支付宝应该如何体现？"
        ]
        embedder = Embedding('solutions/antllm/glm-super-mini-model')
        embeddings = embedder.get_embedding(texts)
        self.assertEqual(embeddings.size()[0], len(texts))
        similarity = torch.cosine_similarity(embeddings[None, ...], embeddings[:, None], dim=-1)
        print(similarity)


if __name__ == '__main__':
    unittest.main()
