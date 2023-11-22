# coding=utf-8
# @Author: zhaohailin.zhl
# @Date: 2023-04-20
import os
from unittest import TestCase, main
from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer


class TestGLMTokenizer(TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(__file__)
        self.tokenizer = GLMTokenizer.from_pretrained(
            os.path.join(self.base_dir, "../..", "zhen_sp5/")
        )

    def is_equal(self, text):
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        recover_text = self.tokenizer.decode(token_ids)
        self.assertEqual(text, recover_text)

    def test_special_tokens(self):
        self.assertTrue(None not in self.tokenizer.all_special_tokens)

    def test_tokenize(self):
        text = "hello beijing, 我爱北京天安门"
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        recover_text = self.tokenizer.decode(token_ids)
        self.assertEqual(text, recover_text)

        token_ids_with_special = self.tokenizer.encode(text, add_special_tokens=True)
        self.assertEqual(
            token_ids_with_special,
            self.tokenizer.convert_tokens_to_ids(["[CLS]"])
            + token_ids
            + self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"]),
        )

    def test_puncts(self):
        text = "，我爱：北京天安门！"
        self.is_equal(text)

    def test_blanks(self):
        # 空格
        text = '    if isinstance(p, Chinese):\n        Say("我爱北京天安门！")'
        self.is_equal(text)

        # tab
        text = '\tif isinstance(p, Chinese):\n\t\tSay("我爱北京天安门！")'
        self.is_equal(text)

        text = "    for i in range(10):"
        self.is_equal(text)

    def test_unk(self):
        text = "😄😭😁  测试锟斤拷"
        self.is_equal(text)


if __name__ == "__main__":
    main()
