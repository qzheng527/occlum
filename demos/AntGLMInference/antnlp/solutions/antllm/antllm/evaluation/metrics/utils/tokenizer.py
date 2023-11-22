#!/usr/bin/env python
# coding=utf-8
# @Author: tianxuan.jl
# @Date: Wed 29 Mar 2023 09:05:13 PM CST
import re
import string   


def tokenize(text):
    chinese_punctuation = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
    punctuations = string.punctuation + chinese_punctuation
    text = text.translate(str.maketrans({key: " {0} ".format(key) for key in punctuations}))
    word_list = []
    prev_pos = -1
    for i in range(len(text)):
        if '\u4e00' <= text[i] <= '\u9fa5':  # 为中文
            if i > prev_pos + 1:
                word_list.append(text[prev_pos+1: i]) # noqa
            word_list.append(text[i])
            prev_pos = i
            continue
    if prev_pos < len(text) - 1:
       word_list.append(text[prev_pos+1:]) # noqa
    return re.sub('\s+', ' ', ' '.join(word_list))


def main():
    text = 'hello, world，我us是中国人test 对吧'
    print(tokenize(text), type(text))


if "__main__" == __name__:
    main()
